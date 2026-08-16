# Demonstration Flow: AI Response Quality Evaluator

This script outlines how to showcase the final platform to stakeholders.

## 1. Introduction (2 mins)
- **Objective:** Briefly explain that the tool automates LLM output quality assurance by using a panel of AI/heuristic judges (Relevance, Accuracy, Completeness).
- **Architecture:** Mention the separation of concerns: FastAPI backend, Streamlit frontend, FAISS RAG grounding, and multi-agent grading.

## 2. Single Response Evaluation (3 mins)
- Navigate to the **"🎯 Single Response Evaluation"** tab.
- Enter a sample Question: "What is the speed of light?"
- Enter a flawed Candidate Response: "The speed of light is 100 miles per hour."
- Enter the Reference Answer: "299,792,458 m/s"
- **Action:** Click "Evaluate Response".
- **Talking Points:** Show how the Verdict Scorecard immediately flags a "Fail" verdict. Expand the detailed Judge Reports below to show the specific hallucinated statement and low Accuracy score.

## 3. Batch Evaluation (4 mins)
- Navigate to the **"📊 Batch Evaluation (CSV)"** tab.
- Explain the scenario: "We have deployed two different AI models (System A and System B). We need to figure out which one is better."
- Upload `batch_examples/ai_system_b_llama3.csv`. Run the evaluation.
- Upload `batch_examples/ai_system_a_gpt4.csv`. Run the evaluation.
- **Talking Points:** Show the tabular outputs, noting how System A gets higher scores and Pass verdicts compared to System B.

## 4. Analytics Dashboard & Reporting (4 mins)
- Navigate to the **"📈 Analytics Dashboard"** tab.
- **Talking Points:**
  - Point out the Key Performance Indicators (Total Evals, Pass Rate, Avg Score, Hallucination Rate).
  - Hover over the Plotly Pie Chart (Verdict Distribution) and Bar Chart (Dimension Averages).
  - Use the Filter dropdown to isolate "Fail" responses.
- **Action:** Click "Generate PDF Report" and open the downloaded PDF.
- **Talking Points:** Show how the PDF neatly structures the batch summary and provides itemized breakdowns of hallucinated claims for easy executive review.

## 5. Conclusion (2 mins)
- Summarize strengths (deterministic, fast, visual, comprehensive).
- Mention future enhancements (swapping heuristic judges for LLM-as-a-judge API calls, adding more evaluation dimensions).
