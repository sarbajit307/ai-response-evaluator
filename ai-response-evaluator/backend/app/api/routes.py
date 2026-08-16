import os
import csv
import json
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.rag.vector_store import FAISS_AVAILABLE, EMBEDDINGS_AVAILABLE
from backend.app.schemas.evaluation import EvaluationResponse, HealthResponse, EvaluationRequest
from backend.app.services.evaluation_service import evaluation_service
from backend.app.services.generation_service import generation_service
from backend.app.rag.pipeline import rag_pipeline
from backend.app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Verifies backend database, vector stores, and configured LLM provider."""
    # Check SQLite connectivity
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    # Check FAISS index loaded
    vector_status = "initialized" if FAISS_AVAILABLE else "mocked (reduced functionality)"

    return HealthResponse(
        status="healthy",
        database=db_status,
        vector_store=vector_status,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.OPENAI_MODEL_NAME if settings.LLM_PROVIDER == "openai" else settings.OLLAMA_MODEL_NAME
    )


@router.post("/upload-reference")
def upload_reference(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    source_name: str = Form("api_upload")
):
    """Adds textual reference documents to the RAG FAISS vector base."""
    if not text and not file:
        raise HTTPException(status_code=400, detail="Either 'text' or 'file' must be supplied.")

    temp_path = None
    chunks_count = 0

    try:
        if file:
            # Create a temporary directory in workspace
            os.makedirs("./temp_uploads", exist_ok=True)
            temp_path = os.path.join("./temp_uploads", file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            chunks_count = rag_pipeline.index_file(temp_path, source_name=file.filename)
        else:
            chunks_count = rag_pipeline.index_text(text, source_name=source_name)

        return {
            "status": "success",
            "message": f"Successfully indexed knowledge source '{file.filename if file else source_name}'",
            "chunks_created": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        # Cleanup temporary uploaded files
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    question: str = Form(...),
    response: str = Form(...),
    reference_answer: Optional[str] = Form(None),
    use_rag: bool = Form(True),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Evaluates a single model response against structured judges, RAG, and reference answers."""
    # If a document is uploaded directly during the evaluation, index it first
    if file:
        os.makedirs("./temp_uploads", exist_ok=True)
        temp_path = os.path.join("./temp_uploads", file.filename)
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            rag_pipeline.index_file(temp_path, source_name=file.filename)
            use_rag = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to index temporary file: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    try:
        eval_record = evaluation_service.evaluate_response(
            db=db,
            question=question,
            response=response,
            reference_answer=reference_answer,
            use_rag=use_rag
        )

        # Parse lists back from JSON strings
        suggestions_list = json.loads(eval_record.suggestions) if eval_record.suggestions else []
        hallucinated_list = json.loads(eval_record.hallucinated_statements) if eval_record.hallucinated_statements else []
        omissions_list = json.loads(eval_record.omissions) if eval_record.omissions else []

        return EvaluationResponse(
            id=eval_record.id,
            question=eval_record.question,
            response=eval_record.response,
            reference_answer=eval_record.reference_answer,
            overall_score=eval_record.overall_score,
            final_grade=eval_record.final_grade,
            final_verdict=eval_record.final_verdict,
            synthesis=eval_record.synthesis,
            suggestions=suggestions_list,
            hallucinated_statements=hallucinated_list,
            omissions=omissions_list,
            created_at=eval_record.created_at,
            agent_outputs=eval_record.agent_outputs,
            retrieved_contexts=eval_record.retrieved_contexts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/batch-evaluate")
def batch_evaluate(
    file: UploadFile = File(...),
    use_rag: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Processes uploaded CSV file containing evaluation batch datasets."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for batch processing.")

    try:
        # Read CSV file contents
        content = file.file.read().decode("utf-8-sig").splitlines()
        reader = csv.DictReader(content)

        # Validate headers
        required_headers = {"question", "response"}
        headers = set(reader.fieldnames or [])
        if not required_headers.issubset(headers):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain at least 'question' and 'response' headers. Found: {reader.fieldnames}"
            )

        results = []
        success_count = 0
        failure_count = 0

        for row in reader:
            q = row.get("question", "").strip()
            r = row.get("response", "").strip()
            ref = row.get("reference_answer", "").strip()
            if not ref:
                ref = row.get("reference", "").strip() # Support reference header fallback
            
            if not q or not r:
                failure_count += 1
                continue

            try:
                eval_record = evaluation_service.evaluate_response(
                    db=db,
                    question=q,
                    response=r,
                    reference_answer=ref if ref else None,
                    use_rag=use_rag
                )
                success_count += 1
                results.append({
                    "id": eval_record.id,
                    "question": q,
                    "overall_score": eval_record.overall_score,
                    "grade": eval_record.final_grade,
                    "final_verdict": eval_record.final_verdict,
                    "status": "success"
                })
            except Exception as e:
                failure_count += 1
                results.append({
                    "question": q,
                    "error": str(e),
                    "status": "failed"
                })

        return {
            "status": "completed",
            "total_processed": success_count + failure_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "evaluations": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process batch CSV: {str(e)}")


@router.get("/results", response_model=List[EvaluationResponse])
def get_results(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieves list of past evaluations scored by the system."""
    records = evaluation_service.get_evaluations(db, skip=skip, limit=limit)
    response_list = []
    for rec in records:
        suggestions_list = json.loads(rec.suggestions) if rec.suggestions else []
        hallucinated_list = json.loads(rec.hallucinated_statements) if rec.hallucinated_statements else []
        omissions_list = json.loads(rec.omissions) if rec.omissions else []

        response_list.append(
            EvaluationResponse(
                id=rec.id,
                question=rec.question,
                response=rec.response,
                reference_answer=rec.reference_answer,
                overall_score=rec.overall_score,
                final_grade=rec.final_grade,
                final_verdict=rec.final_verdict,
                synthesis=rec.synthesis,
                suggestions=suggestions_list,
                hallucinated_statements=hallucinated_list,
                omissions=omissions_list,
                created_at=rec.created_at,
                agent_outputs=rec.agent_outputs,
                retrieved_contexts=rec.retrieved_contexts
            )
        )
    return response_list


@router.post("/generate-answer")
def generate_answer(
    question: str = Form(...)
):
    """Retrieves relevant reference chunks and generates a grounded response."""
    try:
        result = generation_service.generate_answer(question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")
