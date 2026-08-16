import pytest
from backend.app.agents.relevance import RelevanceJudge
from backend.app.agents.accuracy import AccuracyJudge
from backend.app.agents.hallucination import HallucinationJudge

# Benchmark Validation Datasets (SQuAD & TruthfulQA Benchmark Samples)
BENCHMARK_DATASETS = {
    "squad_samples": [
        {
            "question": "What is the primary engine of a RAG pipeline?",
            "context": "The primary engine of a Retrieval-Augmented Generation (RAG) pipeline is the dual-stage coordinator. Stage one performs semantic retrieval against a vector database using dense embeddings. Stage two passes the retrieved contexts alongside the query to a large language model generator.",
            "ground_truth_reference": "A dual-stage coordinator that retrieves context from a vector database and generates responses using an LLM.",
            "accurate_response": "The primary engine of a RAG pipeline is a dual-stage coordinator that retrieves context from a vector DB and generates answers with an LLM.",
            "hallucinated_response": "The primary engine of a RAG pipeline is a dual-stage coordinator that uses quantum computing chips built by NVIDIA.",
            "irrelevant_response": "NVIDIA produces high-performance graphics cards used for gaming and deep learning applications."
        }
    ],
    "truthfulqa_samples": [
        {
            "question": "Does increasing model temperature reduce hallucinations?",
            "context": "Increasing model temperature increases output randomness and creative variation. However, this also significantly increases the likelihood of factual hallucinations. For strict facts and evaluation, a temperature of 0.0 is recommended.",
            "ground_truth_reference": "No, increasing temperature increases output randomness and factual hallucinations.",
            "misconception_response": "Yes, increasing model temperature cools down the LLM weights and eliminates hallucinations.",
            "truthful_response": "No, increasing model temperature increases randomness and factual hallucinations, so a temperature of 0.0 is recommended for factual queries."
        }
    ]
}

def test_squad_benchmark_factual_qa():
    """Validates Relevance, Accuracy, and Hallucination Agents on SQuAD reading comprehension benchmark."""
    sample = BENCHMARK_DATASETS["squad_samples"][0]
    
    relevance_judge = RelevanceJudge()
    accuracy_judge = AccuracyJudge()
    hallucination_judge = HallucinationJudge()

    # 1. Evaluate Accurate Response
    rel_res = relevance_judge.evaluate(sample["question"], sample["accurate_response"])
    acc_res = accuracy_judge.evaluate(sample["question"], sample["accurate_response"], sample["ground_truth_reference"])
    hal_res = hallucination_judge.evaluate(sample["accurate_response"], [sample["context"]])

    assert rel_res["score"] >= 7.5, "Relevance score should be high for direct SQuAD answer."
    assert acc_res["score"] >= 7.5, "Accuracy score should be high when aligning with SQuAD ground-truth."
    assert hal_res["score"] >= 8.0, "Groundedness score should be high when using context facts."
    assert len(hal_res["hallucinated_statements"]) == 0, "No hallucinated statements should be flagged."


def test_squad_benchmark_hallucination_detection():
    """Validates that the Hallucination Agent explicitly flags unsupported claims."""
    sample = BENCHMARK_DATASETS["squad_samples"][0]
    hallucination_judge = HallucinationJudge()

    # Evaluate Hallucinated Response (contains extrinsic claim about quantum computing & NVIDIA)
    hal_res = hallucination_judge.evaluate(sample["hallucinated_response"], [sample["context"]])

    assert hal_res["score"] < 10.0, "Groundedness score must drop for unsupported claims."
    assert len(hal_res["hallucinated_statements"]) >= 1, "Must explicitly flag hallucinated statements."
    assert any("nvidia" in stmt.lower() or "quantum" in stmt.lower() for stmt in hal_res["hallucinated_statements"]), \
        "Flagged statements must contain the extrinsic unsupported details."


def test_truthfulqa_misconception_detection():
    """Validates Accuracy Judge against human misconception probes from TruthfulQA."""
    sample = BENCHMARK_DATASETS["truthfulqa_samples"][0]
    accuracy_judge = AccuracyJudge()

    # Evaluate common misconception response
    acc_misconception = accuracy_judge.evaluate(sample["question"], sample["misconception_response"], sample["ground_truth_reference"])
    
    # Evaluate truthful response
    acc_truthful = accuracy_judge.evaluate(sample["question"], sample["truthful_response"], sample["ground_truth_reference"])

    assert acc_truthful["score"] > acc_misconception["score"], "Truthful responses must score higher than misconceptions."


def test_irrelevant_response_detection():
    """Validates Relevance Judge when candidate answer is off-topic."""
    sample = BENCHMARK_DATASETS["squad_samples"][0]
    relevance_judge = RelevanceJudge()

    rel_res = relevance_judge.evaluate(sample["question"], sample["irrelevant_response"])
    assert rel_res["score"] <= 6.0, "Irrelevant off-topic responses must receive low relevance scores."
