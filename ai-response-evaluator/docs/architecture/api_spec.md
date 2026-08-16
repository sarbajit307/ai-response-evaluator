# REST API Specification

This document provides detailed schema descriptions and example requests/responses for the **AI Response Quality Evaluator Agent** FastAPI backend.

---

## Base Configuration
* **Default URL**: `http://127.0.0.1:8000`
* **Content Type**: `application/json` (unless specifying multipart uploads for files)

---

## Endpoints

### 1. System Health
Verify backend components.

* **Route**: `GET /health`
* **Response `200 OK`**:
```json
{
  "status": "healthy",
  "database": "connected",
  "vector_store": "initialized",
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini"
}
```

---

### 2. Single Response Evaluation
Trigger a full evaluation run on a candidate response.

* **Route**: `POST /evaluate`
* **Request Format**: `multipart/form-data`
* **Parameters**:
  * `question` (string, Required): The user query prompt.
  * `response` (string, Required): The model response.
  * `reference_answer` (string, Optional): Human reference ground truth.
  * `use_rag` (boolean, Optional, default `true`): Index query to search contexts.
  * `file` (binary, Optional): PDF/TXT reference file to index dynamically for this run.

* **Response `200 OK`**:
```json
{
  "id": 12,
  "question": "What is the primary engine of a RAG pipeline?",
  "response": "The primary engine is the dual-stage coordinator. Stage 1 retrieves context, and Stage 2 generates the answer.",
  "reference_answer": "RAG has a retrieval step followed by LLM response generation.",
  "overall_score": 9.5,
  "final_grade": "A (Excellent)",
  "suggestions": [
    "Maintain current formatting style, focusing on detail completeness."
  ],
  "created_at": "2026-07-12T12:00:00.000000",
  "agent_outputs": [
    {
      "agent_name": "Relevance",
      "score": 10.0,
      "confidence": 0.9,
      "reasoning": "Directly answers the primary question details."
    },
    {
      "agent_name": "Accuracy",
      "score": 9.0,
      "confidence": 0.85,
      "reasoning": "Fully matches factual targets from ground-truth answer."
    },
    {
      "agent_name": "Hallucination",
      "score": 10.0,
      "confidence": 0.95,
      "reasoning": "Perfect grounding, zero unsupported extrinsic facts."
    },
    {
      "agent_name": "Completeness",
      "score": 9.0,
      "confidence": 0.8,
      "reasoning": "Covers all segments of the query prompt."
    }
  ],
  "retrieved_contexts": [
    {
      "content": "The primary engine of a Retrieval-Augmented Generation (RAG) pipeline is the dual-stage coordinator...",
      "source": "squad_bootstrap",
      "score": 0.082
    }
  ]
}
```

---

### 3. Batch Evaluation
Evaluate multiple responses in bulk by uploading a CSV.

* **Route**: `POST /batch-evaluate`
* **Request Format**: `multipart/form-data`
* **Parameters**:
  * `file` (binary, Required): CSV file containing `question`, `response`, and optional `reference_answer` columns.
  * `use_rag` (boolean, Optional, default `true`): Check contexts for each row.

* **Response `200 OK`**:
```json
{
  "status": "completed",
  "total_processed": 2,
  "success_count": 2,
  "failure_count": 0,
  "evaluations": [
    {
      "id": 13,
      "question": "What is FAISS?",
      "score": 9.2,
      "grade": "A (Excellent)",
      "status": "success"
    },
    {
      "id": 14,
      "question": "Explain RAG.",
      "score": 8.7,
      "grade": "B (Good)",
      "status": "success"
    }
  ]
}
```

---

### 4. Upload Reference Source
Index text or files into the global FAISS knowledge base.

* **Route**: `POST /upload-reference`
* **Request Format**: `multipart/form-data`
* **Parameters**:
  * `text` (string, Optional): Raw text segment.
  * `file` (binary, Optional): PDF/TXT document.
  * `source_name` (string, Optional, default `api_upload`): Source identifier tag.

* **Response `200 OK`**:
```json
{
  "status": "success",
  "message": "Successfully indexed knowledge source 'knowledge.txt'",
  "chunks_created": 4
}
```

---

### 5. Fetch Past Evaluations
Query past runs stored in SQLite.

* **Route**: `GET /results`
* **Parameters**:
  * `skip` (integer, Optional, default `0`): Pagination offset.
  * `limit` (integer, Optional, default `100`): Max records.

* **Response `200 OK`**:
Returns an array of `EvaluationResponse` items matching the structure of `POST /evaluate`.
