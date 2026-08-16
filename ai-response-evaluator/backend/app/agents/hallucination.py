from typing import Dict, Any, List
from backend.app.agents.base import BaseJudgeAgent

class HallucinationJudge(BaseJudgeAgent):
    def __init__(self):
        super().__init__(
            name="Hallucination Judge",
            description="Evaluates if the response contains claims unsupported by the retrieved context (groundedness/faithfulness) and flags hallucinated statements."
        )

    def evaluate(self, response: str, contexts: List[str]) -> Dict[str, Any]:
        if not contexts:
            return {
                "score": 10.0,
                "reasoning": "No reference context or source document provided. Grounding verification skipped (assumed perfect).",
                "confidence": 1.0,
                "hallucinated_statements": []
            }

        merged_context = "\n---\n".join(contexts)

        system_prompt = (
            "You are an expert AI Hallucination and Groundedness Judge. Your goal is to evaluate if a generated response "
            "is strictly faithful to and supported by the retrieved source context. Any statements, claims, or details "
            "in the response that cannot be verified directly from the source context are considered hallucinations.\n\n"
            "Score from 0.0 to 10.0, where:\n"
            "- 9.0-10.0: Perfect groundedness; every statement in the response is explicitly supported by the context.\n"
            "- 7.0-8.9: Highly faithful, but contains minor conversational inferences or extrapolations that are harmless.\n"
            "- 5.0-6.9: Moderate hallucinations; some factual claims cannot be verified from the text.\n"
            "- 1.0-4.9: Severe hallucinations; response makes major claims directly contradicted or unsupported by the context.\n"
            "- 0.0: Completely ungrounded, fabricated, or empty response.\n\n"
            "Output your findings in JSON format with exactly four fields:\n"
            "{\n"
            "  \"score\": float,\n"
            "  \"reasoning\": \"string explaining the score\",\n"
            "  \"confidence\": float (0.0 to 1.0),\n"
            "  \"hallucinated_statements\": [\"unsupported claim 1\", \"unsupported claim 2\"]\n"
            "}"
        )

        user_prompt = f"Retrieved Context:\n{merged_context}\n\nCandidate Response:\n{response}"
        res = self.execute_llm(system_prompt, user_prompt)
        if "hallucinated_statements" not in res:
            res["hallucinated_statements"] = []
        return res

    def mock_evaluate(self, user_prompt: str) -> Dict[str, Any]:
        # Parse inputs from text prompt
        parts = user_prompt.split("\n\nCandidate Response:\n")
        context_part = parts[0].replace("Retrieved Context:\n", "").strip()
        response = parts[1].strip() if len(parts) > 1 else ""

        if not context_part or context_part.startswith("No reference context"):
            return {
                "score": 10.0,
                "reasoning": "[Mock Heuristics] Groundedness verification is skipped because retrieved context is empty.",
                "confidence": 1.0,
                "hallucinated_statements": []
            }

        if not response:
            return {
                "score": 0.0,
                "reasoning": "[Mock Heuristics] Response is empty.",
                "confidence": 0.95,
                "hallucinated_statements": []
            }

        # Calculate semantic matching of response sentences against context
        sentences = [s.strip() for s in response.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        
        supported_count = 0
        hallucinated_statements = []
        context_lower = context_part.lower()

        for sent in sentences:
            sent_words = [w for w in sent.lower().split() if len(w) > 3]
            if not sent_words:
                supported_count += 1
                continue
                
            # Check if majority of important words in the sentence appear in the context
            match_count = sum(1 for w in sent_words if w in context_lower)
            match_ratio = match_count / len(sent_words)
            
            if match_ratio >= 0.5: # 50% keyword presence counts as supported in mock heuristics
                supported_count += 1
            else:
                hallucinated_statements.append(sent)

        total_sentences = max(len(sentences), 1)
        groundedness_ratio = supported_count / total_sentences
        score = groundedness_ratio * 10.0

        if score >= 9.0:
            reasoning = f"[Mock Heuristics] Response is highly faithful. All {supported_count}/{total_sentences} sentences are grounded in the source text."
        elif score >= 6.5:
            reasoning = (
                f"[Mock Heuristics] Response is mostly faithful. {supported_count} out of {total_sentences} statements "
                f"are backed by context keywords. Found {len(hallucinated_statements)} unverified statements."
            )
        else:
            reasoning = (
                f"[Mock Heuristics] Low groundedness score. Only {supported_count}/{total_sentences} statements "
                f"align with the context. The candidate response contains heavy extrinsic facts or hallucinations."
            )

        return {
            "score": round(score, 1),
            "reasoning": reasoning,
            "confidence": 0.75,
            "hallucinated_statements": hallucinated_statements
        }
