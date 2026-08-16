from typing import Dict, Any, Optional
from backend.app.agents.base import BaseJudgeAgent

class AccuracyJudge(BaseJudgeAgent):
    def __init__(self):
        super().__init__(
            name="Accuracy Judge",
            description="Evaluates if the response is factually accurate when compared against a trusted reference answer."
        )

    def evaluate(self, question: str, response: str, reference_answer: Optional[str]) -> Dict[str, Any]:
        if not reference_answer:
            return {
                "score": 10.0,
                "reasoning": "No reference answer provided for comparison. Accuracy is assumed perfect by default.",
                "confidence": 1.0
            }

        system_prompt = (
            "You are an expert AI Response Accuracy Judge. Your goal is to evaluate if a generated response "
            "aligns factually with the provided ground-truth reference answer.\n\n"
            "Score from 0.0 to 10.0, where:\n"
            "- 9.0-10.0: Perfect alignment, no factual contradictions, captures all primary facts from the reference.\n"
            "- 7.0-8.9: Minor factual details missing or slightly incomplete coverage, but zero contradictions.\n"
            "- 5.0-6.9: Factual errors on minor points, or highly vague statements that partially match the reference.\n"
            "- 1.0-4.9: Severe factual contradictions or matches almost none of the reference facts.\n"
            "- 0.0: Completely false, opposite claims, or blank answer.\n\n"
            "Output your findings in JSON format with exactly three fields:\n"
            "{\n"
            "  \"score\": float,\n"
            "  \"reasoning\": \"string explaining the score\",\n"
            "  \"confidence\": float (0.0 to 1.0)\n"
            "}"
        )

        user_prompt = f"Question: {question}\n\nResponse: {response}\n\nReference Answer: {reference_answer}"
        return self.execute_llm(system_prompt, user_prompt)

    def mock_evaluate(self, user_prompt: str) -> Dict[str, Any]:
        # Parse inputs from text prompt
        lines = user_prompt.split("\n")
        response = ""
        reference = ""
        for line in lines:
            if line.startswith("Response:"):
                response = line.replace("Response:", "").strip()
            elif line.startswith("Reference Answer:"):
                reference = line.replace("Reference Answer:", "").strip()

        if not reference:
            return {
                "score": 10.0,
                "reasoning": "[Mock Heuristics] No reference answer provided for accuracy verification.",
                "confidence": 1.0
            }

        if not response:
            return {
                "score": 0.0,
                "reasoning": "[Mock Heuristics] Candidate response is empty.",
                "confidence": 0.95
            }

        # Calculate keyword overlap between candidate response and reference answer
        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())

        overlap = len(ref_words.intersection(resp_words))
        ratio = overlap / max(len(ref_words), 1)

        # Baseline scoring
        score = 3.0 + (ratio * 7.0)
        score = min(score, 10.0)

        # Look for negative modifiers (e.g. "not", "no", "never", "cannot") to catch contradictions
        negators = {"not", "no", "never", "cannot", "incorrect", "false", "contradict"}
        resp_negators = resp_words.intersection(negators)
        ref_negators = ref_words.intersection(negators)
        
        # If negator density differs, flag possible contradiction
        if len(resp_negators) != len(ref_negators):
            score = max(2.0, score - 2.5)
            reasoning = (
                f"[Mock Heuristics] Detected potential logical negation mismatch between response and reference answer. "
                f"Word overlap ratio: {ratio:.2f}. Score penalized for possible contradiction."
            )
        else:
            reasoning = (
                f"[Mock Heuristics] Factual alignment is strong. The candidate response successfully matches "
                f"{ratio*100:.1f}% of key terms from the ground-truth reference answer."
            )

        return {
            "score": round(score, 1),
            "reasoning": reasoning,
            "confidence": 0.85
        }
