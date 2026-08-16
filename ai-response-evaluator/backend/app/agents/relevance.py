from typing import Dict, Any
from backend.app.agents.base import BaseJudgeAgent

class RelevanceJudge(BaseJudgeAgent):
    def __init__(self):
        super().__init__(
            name="Relevance Judge",
            description="Evaluates if the response directly addresses the user question and avoids fluff."
        )

    def evaluate(self, question: str, response: str) -> Dict[str, Any]:
        system_prompt = (
            "You are an expert AI Response Relevance Judge. Your goal is to evaluate if a generated response "
            "directly addresses the user's question. A relevant answer must answer the question directly, "
            "avoid tangents, and contain no redundant padding.\n\n"
            "Score from 0.0 to 10.0, where:\n"
            "- 9.0-10.0: Perfectly relevant, concise, directly addresses the prompt.\n"
            "- 7.0-8.9: Highly relevant, but contains minor conversational fluff or slight wordiness.\n"
            "- 5.0-6.9: Partially relevant, talks about the correct general topic but misses the core query.\n"
            "- 1.0-4.9: Mostly irrelevant or contains heavy tangents.\n"
            "- 0.0: Completely irrelevant, gibberish, or empty.\n\n"
            "Output your findings in JSON format with exactly three fields:\n"
            "{\n"
            "  \"score\": float,\n"
            "  \"reasoning\": \"string explaining the score\",\n"
            "  \"confidence\": float (0.0 to 1.0)\n"
            "}"
        )

        user_prompt = f"Question: {question}\n\nResponse: {response}"
        return self.execute_llm(system_prompt, user_prompt)

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

        # Heuristic calculations for mock relevance
        q_words = set(question.lower().split())
        r_words = set(response.lower().split())

        if not response or len(r_words) < 3:
            return {
                "score": 0.0,
                "reasoning": "[Mock Heuristics] The generated response is empty or too short to be relevant.",
                "confidence": 0.95
            }

        # Calculate word overlap ratio
        overlap = len(q_words.intersection(r_words))
        ratio = overlap / max(len(q_words), 1)

        # Baseline score: if there's any length, start with a 5.0. If words match, increase it.
        score = 5.0 + (ratio * 5.0)
        score = min(score, 10.0)

        # Penalize extreme lengths if question is very short (possible verbosity fluff)
        if len(response) > 1000 and len(question) < 50:
            score -= 1.0
            reasoning = (
                f"[Mock Heuristics] The response is highly verbose ({len(response)} chars) relative to a short query. "
                f"Topic keywords overlap moderately. Score adjusted for potential conversational fluff."
            )
        else:
            reasoning = (
                f"[Mock Heuristics] The response addresses the general query topics with a matching vocabulary ratio of "
                f"{ratio:.2f}. Structured text flows naturally."
            )

        return {
            "score": round(score, 1),
            "reasoning": reasoning,
            "confidence": 0.8
        }
