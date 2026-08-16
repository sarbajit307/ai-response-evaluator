import json
from typing import Dict, Any, List
from backend.app.agents.base import BaseJudgeAgent

class VerdictAgent(BaseJudgeAgent):
    def __init__(self):
        super().__init__(
            name="Verdict Agent",
            description="Synthesizes all individual judge scores and generates a master scorecard with detailed recommendations, overall verdict and synthesis."
        )

    def evaluate_verdict(self, question: str, response: str, scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates final scores and invokes the LLM (or mock) to compile recommendations, grade, and verdict."""
        # Calculate mathematical weighted score
        rel = scores.get("relevance", {}).get("score", 5.0)
        acc = scores.get("accuracy", {}).get("score", 5.0)
        hal = scores.get("hallucination", {}).get("score", 5.0)
        comp = scores.get("completeness", {}).get("score", 5.0)

        weighted_score = (0.25 * rel) + (0.25 * acc) + (0.30 * hal) + (0.20 * comp)
        weighted_score = round(weighted_score, 2)

        # Map score to grade
        if weighted_score >= 9.0:
            grade = "A (Excellent)"
        elif weighted_score >= 7.5:
            grade = "B (Good)"
        elif weighted_score >= 5.5:
            grade = "C (Fair)"
        else:
            grade = "F (Fail)"

        # Map score to quality verdict (Pass/Needs Improvement/Fail)
        if weighted_score >= 7.5:
            verdict_status = "Pass"
        elif weighted_score >= 5.0:
            verdict_status = "Needs Improvement"
        else:
            verdict_status = "Fail"

        system_prompt = (
            "You are an expert AI Response Verdict Agent. Your goal is to review the individual agent evaluation report "
            "for a candidate response and compile a master synthesis (a consolidated reasoning summary explaining the verdict), "
            "strengths, weaknesses, and clear actionable suggestions for improvement.\n\n"
            "Output your findings in JSON format with exactly two fields:\n"
            "{\n"
            "  \"synthesis\": \"a paragraph summarizing quality findings and acts as a consolidated reasoning summary\",\n"
            "  \"suggestions\": [\n"
            "    \"suggestion 1\",\n"
            "    \"suggestion 2\"\n"
            "  ]\n"
            "}"
        )

        scores_summary_text = json.dumps(scores, indent=2)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Candidate Response: {response}\n\n"
            f"Evaluations Report:\n{scores_summary_text}\n\n"
            f"Calculated Weighted Score: {weighted_score}\n"
            f"Calculated Grade: {grade}\n"
            f"Calculated Verdict: {verdict_status}"
        )

        verdict_data = self.execute_llm(system_prompt, user_prompt)
        
        # Merge calculated fields into verdict data
        verdict_data["overall_score"] = weighted_score
        verdict_data["final_grade"] = grade
        verdict_data["final_verdict"] = verdict_status

        return verdict_data

    def mock_evaluate(self, user_prompt: str) -> Dict[str, Any]:
        # Fallback compiler parsing the text prompt
        lines = user_prompt.split("\n")
        weighted_score = 5.0
        grade = "C (Fair)"
        verdict_status = "Needs Improvement"
        
        # Parse final calculated details from prompt
        for line in lines:
            if line.startswith("Calculated Weighted Score:"):
                weighted_score = float(line.replace("Calculated Weighted Score:", "").strip())
            elif line.startswith("Calculated Grade:"):
                grade = line.replace("Calculated Grade:", "").strip()
            elif line.startswith("Calculated Verdict:"):
                verdict_status = line.replace("Calculated Verdict:", "").strip()

        # Generate rule-based recommendations based on scores
        suggestions = []
        if weighted_score < 7.5:
            suggestions.append("Clarify the core terminology to prevent semantic ambiguity.")
        if weighted_score < 9.0:
            suggestions.append("Structure the output using lists or bullet points to improve reading accessibility.")
            suggestions.append("Ensure every factual claim references explicit details from source documentation.")
        else:
            suggestions.append("Maintain current formatting style, focusing on detail completeness.")

        synthesis = (
            f"[Mock Verdict] The response achieved a weighted score of {weighted_score} earning a grade of {grade} "
            f"and a quality verdict of '{verdict_status}'. Evaluations show stable structure with minor spaces "
            f"for context verification refinement."
        )

        return {
            "synthesis": synthesis,
            "suggestions": suggestions,
            "overall_score": weighted_score,
            "final_grade": grade,
            "final_verdict": verdict_status
        }
