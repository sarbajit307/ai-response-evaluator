import json
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from backend.app.database.models import Evaluation, AgentOutput, RetrievedContext
from backend.app.rag.pipeline import rag_pipeline
from backend.app.agents.relevance import RelevanceJudge
from backend.app.agents.accuracy import AccuracyJudge
from backend.app.agents.hallucination import HallucinationJudge
from backend.app.agents.completeness import CompletenessJudge
from backend.app.agents.verdict import VerdictAgent

class EvaluationService:
    def __init__(self):
        # Initialize judges
        self.relevance_judge = RelevanceJudge()
        self.accuracy_judge = AccuracyJudge()
        self.hallucination_judge = HallucinationJudge()
        self.completeness_judge = CompletenessJudge()
        self.verdict_agent = VerdictAgent()

    def evaluate_response(
        self,
        db: Session,
        question: str,
        response: str,
        reference_answer: Optional[str] = None,
        use_rag: bool = True
    ) -> Evaluation:
        """Runs the multi-agent evaluation pipeline and stores the scorecard."""
        # 1. Retrieve RAG Context
        retrieved_contexts = []
        context_texts = []
        if use_rag:
            # Query FAISS index for matching contexts
            retrieved = rag_pipeline.retrieve_context(question, k=3)
            for idx, ctx in enumerate(retrieved):
                retrieved_contexts.append(
                    RetrievedContext(
                        content=ctx["content"],
                        source=ctx["source"],
                        score=ctx["score"]
                    )
                )
                context_texts.append(ctx["content"])
        
        # If no RAG contexts retrieved but reference answer is provided, use reference as grounding
        if not context_texts and reference_answer:
            context_texts.append(reference_answer)

        # 2. Run Individual Agents in sequence (parallel simulated)
        scores = {}
        
        # Relevance
        rel_res = self.relevance_judge.evaluate(question, response)
        scores["relevance"] = rel_res

        # Accuracy
        acc_res = self.accuracy_judge.evaluate(question, response, reference_answer)
        scores["accuracy"] = acc_res

        # Hallucination/Groundedness
        hal_res = self.hallucination_judge.evaluate(response, context_texts)
        scores["hallucination"] = hal_res

        # Completeness
        comp_res = self.completeness_judge.evaluate(question, response)
        scores["completeness"] = comp_res

        # 3. Compile Master Verdict
        verdict = self.verdict_agent.evaluate_verdict(question, response, scores)

        # 4. Save to Database
        # Convert lists to JSON string for SQLite storage
        suggestions_str = json.dumps(verdict.get("suggestions", []))
        hallucinated_statements_str = json.dumps(hal_res.get("hallucinated_statements", []))
        omissions_str = json.dumps(comp_res.get("omissions", []))
        
        eval_record = Evaluation(
            question=question,
            response=response,
            reference_answer=reference_answer,
            overall_score=verdict["overall_score"],
            final_grade=verdict["final_grade"],
            final_verdict=verdict.get("final_verdict", "Needs Improvement"),
            suggestions=suggestions_str,
            hallucinated_statements=hallucinated_statements_str,
            omissions=omissions_str,
            synthesis=verdict.get("synthesis", "")
        )
        
        db.add(eval_record)
        db.commit()
        db.refresh(eval_record)

        # Associate agent outputs
        agent_outputs = [
            AgentOutput(
                evaluation_id=eval_record.id,
                agent_name="Relevance",
                score=rel_res["score"],
                confidence=rel_res["confidence"],
                reasoning=rel_res["reasoning"]
            ),
            AgentOutput(
                evaluation_id=eval_record.id,
                agent_name="Accuracy",
                score=acc_res["score"],
                confidence=acc_res["confidence"],
                reasoning=acc_res["reasoning"]
            ),
            AgentOutput(
                evaluation_id=eval_record.id,
                agent_name="Hallucination",
                score=hal_res["score"],
                confidence=hal_res["confidence"],
                reasoning=hal_res["reasoning"]
            ),
            AgentOutput(
                evaluation_id=eval_record.id,
                agent_name="Completeness",
                score=comp_res["score"],
                confidence=comp_res["confidence"],
                reasoning=comp_res["reasoning"]
            )
        ]

        db.add_all(agent_outputs)

        # Associate retrieved contexts
        for ctx in retrieved_contexts:
            ctx.evaluation_id = eval_record.id
            db.add(ctx)

        db.commit()
        db.refresh(eval_record)

        return eval_record

    def get_evaluations(self, db: Session, skip: int = 0, limit: int = 100) -> List[Evaluation]:
        """Fetch past evaluations from DB."""
        return db.query(Evaluation).order_by(Evaluation.created_at.desc()).offset(skip).limit(limit).all()

    def get_evaluation(self, db: Session, evaluation_id: int) -> Optional[Evaluation]:
        """Fetch a specific evaluation scorecard."""
        return db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()

# Instantiate singleton
evaluation_service = EvaluationService()
