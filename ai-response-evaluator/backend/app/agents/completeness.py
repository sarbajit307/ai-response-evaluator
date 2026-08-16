from typing import Dict, Any, List
from backend.app.agents.base import BaseJudgeAgent

class CompletenessJudge(BaseJudgeAgent):
    def __init__(self):
        super().__init__(
            name="Completeness Judge",
            description="Evaluates if the response addresses all implicit and explicit criteria of the prompt and identifies omissions."
        )

    def evaluate(self, question: str, response: str) -> Dict[str, Any]:
        system_prompt = (
            "You are an expert AI Response Completeness Judge. Your goal is to evaluate if a generated response "
            "fully answers all implicit and explicit sub-questions or requirements contained in the user's prompt.\n\n"
            "Score from 0.0 to 10.0, where:\n"
            "- 9.0-10.0: Fully comprehensive; answers all parts of the question, provides clear structured details.\n"
            "- 7.0-8.9: Good completeness, answers primary queries but misses minor secondary points or formatting constraints.\n"
            "- 5.0-6.9: Average completeness, answers only the main question but leaves many implicit elements unaddressed.\n"
            "- 1.0-4.9: Incomplete; misses major parts of the question, very brief or half-finished answers.\n"
            "- 0.0: Completely empty, zero information, or unrelated response.\n\n"
            "Output your findings in JSON format with exactly four fields:\n"
            "{\n"
            "  \"score\": float,\n"
            "  \"reasoning\": \"string explaining the score\",\n"
            "  \"confidence\": float (0.0 to 1.0),\n"
            "  \"omissions\": [\"omitted detail 1\", \"omitted detail 2\"]\n"
            "}"
        )

        user_prompt = f"Question: {question}\n\nResponse: {response}"
        res = self.execute_llm(system_prompt, user_prompt)
        if "omissions" not in res:
            res["omissions"] = []
        return res

    def mock_evaluate(self, user_prompt: str) -> Dict[str, Any]:
        # Parse inputs from text prompt
        lines = user_prompt.split("\n")
        question = ""
        response = ""
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Response:"):
                response = line.replace("Response:", "").strip()

        if not response:
            return {
                "score": 0.0,
                "reasoning": "[Mock Heuristics] Response is empty.",
                "confidence": 0.95,
                "omissions": ["Entire response is missing."]
            }

        # Analyze question complexity by checking for coordinators (and, or, also, list, steps, explain, why)
        indicators = ["and", "also", "explain", "why", "list", "steps", "how", "compare"]
        question_lower = question.lower()
        complexity = sum(1 for ind in indicators if ind in question_lower)

        # Check response details (length and formatting elements)
        word_count = len(response.split())
        has_formatting = int("-" in response or "*" in response or "\n" in response or "1." in response)

        # Base scoring based on word count
        if word_count > 100:
            score = 8.5
        elif word_count > 40:
            score = 7.0
        else:
            score = 4.5

        # Add formatting bonus
        if has_formatting:
            score += 1.0

        # Determine omissions
        omissions = []
        if complexity >= 2 and word_count < 50:
            score = max(2.0, score - 2.5)
            reasoning = (
                f"[Mock Heuristics] The prompt is complex (implied multiple sub-questions), but the response is "
                f"brief ({word_count} words). Fails to provide comprehensive explanations."
            )
            omissions.append("Did not cover all implicit sub-questions of the prompt.")
        else:
            reasoning = (
                f"[Mock Heuristics] The response has a length of {word_count} words and structures the answer "
                f"with visual elements. Captures primary requirements adequately."
            )
            
        if not has_formatting:
            omissions.append("Lacks structured sections or bullet points (e.g. lists/steps) to organize the information.")

        score = min(score, 10.0)

        return {
            "score": round(score, 1),
            "reasoning": reasoning,
            "confidence": 0.8,
            "omissions": omissions
        }
