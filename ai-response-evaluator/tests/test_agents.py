import pytest
from backend.app.agents.relevance import RelevanceJudge
from backend.app.agents.accuracy import AccuracyJudge
from backend.app.agents.hallucination import HallucinationJudge
from backend.app.agents.completeness import CompletenessJudge
from backend.app.agents.verdict import VerdictAgent

def test_relevance_judge_logic():
    agent = RelevanceJudge()
    
    # Assert mock execution
    res = agent.evaluate(
        question="What is the chemical symbol for gold?",
        response="The chemical symbol for gold is Au. Gold is element 79."
    )
    assert "score" in res
    assert "reasoning" in res
    assert res["score"] >= 8.0 # Highly relevant words matched

    empty_res = agent.evaluate(
        question="Explain mitosis.",
        response=""
    )
    assert empty_res["score"] == 0.0


def test_accuracy_judge_logic():
    agent = AccuracyJudge()
    
    res = agent.evaluate(
        question="What is 5 + 5?",
        response="The sum of five and five is ten.",
        reference_answer="The answer is 10."
    )
    assert res["score"] >= 6.0 # Factual alignment matching 'ten' and '10' words

    no_ref_res = agent.evaluate(
        question="What is 5 + 5?",
        response="10.",
        reference_answer=""
    )
    assert no_ref_res["score"] == 10.0 # Assumed perfect if no reference is given


def test_hallucination_judge_logic():
    agent = HallucinationJudge()
    
    contexts = ["The capital of Japan is Tokyo. Tokyo is highly populated."]
    
    # Grounded response
    res = agent.evaluate(
        response="Tokyo is the capital of Japan.",
        contexts=contexts
    )
    assert res["score"] >= 8.0

    # Hallucinated response
    hallucinated_res = agent.evaluate(
        response="Tokyo is the capital of Japan and has the best sushi restaurants globally.",
        contexts=contexts
    )
    assert hallucinated_res["score"] < 10.0 # sushi is extrinsic claim


def test_completeness_judge_logic():
    agent = CompletenessJudge()
    
    # Detailed response
    res = agent.evaluate(
        question="List three primary states of matter.",
        response="- Solid: Fixed volume and shape.\n- Liquid: Fixed volume, variable shape.\n- Gas: Variable volume and shape."
    )
    assert res["score"] >= 5.5 # List structure + length matches

    # Too brief response
    brief_res = agent.evaluate(
        question="List three primary states of matter and explain their molecular properties.",
        response="Solid, liquid, and gas."
    )
    assert brief_res["score"] < 8.0 # Brief response to complex query penalized


def test_verdict_agent_synthesis():
    agent = VerdictAgent()
    
    scores = {
        "relevance": {"score": 9.0, "reasoning": "Direct answer", "confidence": 0.9},
        "accuracy": {"score": 9.5, "reasoning": "Correct facts", "confidence": 0.95},
        "hallucination": {"score": 10.0, "reasoning": "Grounded", "confidence": 1.0},
        "completeness": {"score": 8.0, "reasoning": "Answers query", "confidence": 0.8}
    }
    
    res = agent.evaluate_verdict(
        question="Who was Albert Einstein?",
        response="Einstein was a theoretical physicist who developed the theory of relativity.",
        scores=scores
    )
    
    assert "overall_score" in res
    assert "final_grade" in res
    assert "synthesis" in res
    assert len(res["suggestions"]) >= 1
    
    # = 2.25 + 2.375 + 3.0 + 1.6 = 9.225 -> rounds to 9.22 in Python (Banker's rounding)
    assert res["overall_score"] == 9.22
    assert "A" in res["final_grade"]
