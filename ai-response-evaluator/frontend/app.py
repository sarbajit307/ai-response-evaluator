import os
import json
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from frontend.report_generator import generate_pdf_report

# Setup Page Configuration
st.set_page_config(
    page_title="AI Response Quality Evaluator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration & Backend Endpoint
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Session State for routing QA Generator answers
if "eval_question" not in st.session_state:
    st.session_state["eval_question"] = ""
if "eval_response" not in st.session_state:
    st.session_state["eval_response"] = ""

# Custom Styles for Sleek Dashboard
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stButton>button {
        background-color: #6366f1 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #4f46e5 !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-a {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-b {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-c {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-f {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for API communication
def get_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def get_history():
    try:
        response = requests.get(f"{BACKEND_URL}/results", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# App Header
st.title("🛡️ AI Response Quality Evaluator Agent")
st.caption("Milestone 1 Production Dashboard — Multi-agent evaluation, RAG validation, and detailed LLM metrics.")

# Sidebar diagnostics & history
st.sidebar.subheader("System Status")
health = get_health()
if health:
    st.sidebar.success("Backend Connected")
    st.sidebar.markdown(f"**Database:** `{health['database']}`")
    st.sidebar.markdown(f"**Vector DB:** `{health['vector_store']}`")
    st.sidebar.markdown(f"**LLM Provider:** `{health['llm_provider']}` (`{health['llm_model']}`)")
else:
    st.sidebar.error("Backend Disconnected")
    st.sidebar.info(f"Start the FastAPI backend server first (running on `{BACKEND_URL}`).")

st.sidebar.divider()

# Load evaluation history
st.sidebar.subheader("Evaluation History")
history_data = get_history()

selected_history = None
if history_data:
    history_options = {
        f"#{h['id']} - {h['question'][:30]}... ({h['overall_score']:.1f}/10)": h 
        for h in history_data
    }
    selected_label = st.sidebar.selectbox("Load previous evaluation:", ["-- Select Run --"] + list(history_options.keys()))
    if selected_label != "-- Select Run --":
        selected_history = history_options[selected_label]
else:
    st.sidebar.text("No history records yet.")

st.sidebar.divider()

# Reference Doc Manager in Sidebar
st.sidebar.subheader("Reference Library")
upload_ref_file = st.sidebar.file_uploader("Index new document (PDF/TXT):", type=["pdf", "txt"], key="ref_library_upload")
if upload_ref_file:
    if st.sidebar.button("Index Document", key="run_indexing"):
        with st.sidebar.spinner("Indexing chunks..."):
            try:
                files = {"file": (upload_ref_file.name, upload_ref_file.getvalue(), upload_ref_file.type)}
                data = {"source_name": upload_ref_file.name}
                res = requests.post(f"{BACKEND_URL}/upload-reference", files=files, data=data)
                if res.status_code == 200:
                    st.sidebar.success(f"Indexed successfully! Chunks: {res.json().get('chunks_created')}")
                    st.rerun()
                else:
                    st.sidebar.error(f"Error: {res.text}")
            except Exception as e:
                st.sidebar.error(f"Failed to communicate with API: {e}")

# Layout tabs
tab_single, tab_batch, tab_generate, tab_analytics = st.tabs(["🎯 Single Response Evaluation", "📊 Batch Evaluation (CSV)", "🤖 RAG QA Generator", "📈 Analytics Dashboard"])

with tab_single:
    st.header("Single Response Evaluation Workbench")
    
    # Check if loaded from history
    is_loaded = selected_history is not None
    loaded_data = selected_history if is_loaded else {}

    col_inputs, col_visuals = st.columns([1, 1])

    with col_inputs:
        st.subheader("Evaluation Inputs")
        
        # User input fields
        question_val = loaded_data.get("question", st.session_state.get("eval_question", ""))
        response_val = loaded_data.get("response", st.session_state.get("eval_response", ""))
        reference_val = loaded_data.get("reference_answer", "")

        inp_question = st.text_area("Question / Prompt:", value=question_val, height=80, placeholder="What is the user asking?")
        inp_response = st.text_area("Candidate Response:", value=response_val, height=180, placeholder="Enter the AI response to evaluate...")
        inp_reference = st.text_area("Reference Answer (Optional):", value=reference_val, height=80, placeholder="Enter the trusted ground-truth answer...")
        
        inp_doc = st.file_uploader("Upload ground-truth context document (PDF/TXT):", type=["pdf", "txt"], help="Will be indexed dynamically for this run.")
        use_rag = st.checkbox("Enable RAG Grounding Search", value=True)

        btn_run = st.button("Evaluate Response", disabled=(not inp_question or not inp_response))

        if btn_run and not is_loaded:
            with st.spinner("Analyzing response through Multi-Agent judging panel..."):
                try:
                    # Prepare request
                    form_data = {
                        "question": inp_question,
                        "response": inp_response,
                        "use_rag": str(use_rag)
                    }
                    if inp_reference:
                        form_data["reference_answer"] = inp_reference
                    
                    files = None
                    if inp_doc:
                        files = {"file": (inp_doc.name, inp_doc.getvalue(), inp_doc.type)}

                    res = requests.post(f"{BACKEND_URL}/evaluate", data=form_data, files=files)
                    
                    if res.status_code == 200:
                        st.success("Evaluation complete!")
                        loaded_data = res.json()
                        is_loaded = True
                        st.rerun() # Refresh to draw details
                    else:
                        st.error(f"Backend Server Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to evaluate: {e}")

    with col_visuals:
        st.subheader("Verdict Scorecard")
        if is_loaded:
            score = loaded_data["overall_score"]
            grade = loaded_data["final_grade"]
            
            # Match badge colors
            badge_class = "badge-f"
            if "A" in grade:
                badge_class = "badge-a"
            elif "B" in grade:
                badge_class = "badge-b"
            elif "C" in grade:
                badge_class = "badge-c"

            # Match badge colors for Verdict
            verdict = loaded_data.get("final_verdict", "Needs Improvement")
            verdict_badge_class = "badge-f"
            if verdict == "Pass":
                verdict_badge_class = "badge-a"
            elif verdict == "Needs Improvement":
                verdict_badge_class = "badge-c"

            # Render scorecard summary
            st.markdown(f"""
            <div class="metric-card">
                <h3>Overall Rating: <span style="font-size: 2rem;">{score:.2f}</span> / 10.0</h3>
                <h4>Verdict: <span class="{verdict_badge_class}">{verdict}</span> (Grade {grade})</h4>
            </div>
            """, unsafe_allow_html=True)

            synthesis = loaded_data.get("synthesis", "")
            if synthesis:
                st.markdown(f"**Consolidated Reasoning Summary:**\n\n{synthesis}")

            # Draw Plotly Radar Chart
            agents_outputs = loaded_data.get("agent_outputs", [])
            if agents_outputs:
                categories = [a["agent_name"] for a in agents_outputs]
                scores = [a["score"] for a in agents_outputs]

                # Close the radar loop
                categories_loop = categories + [categories[0]]
                scores_loop = scores + [scores[0]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=scores_loop,
                    theta=categories_loop,
                    fill='toself',
                    fillcolor='rgba(99, 102, 241, 0.2)',
                    line=dict(color='#6366f1', width=2),
                    name='Agent Scores'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10]),
                        bgcolor='rgba(11, 15, 25, 0.8)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f3f4f6',
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=280
                )
                st.plotly_chart(fig, use_container_width=True)

            # Export options
            st.subheader("Export Results")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                json_str = json.dumps(loaded_data, indent=2)
                st.download_button(
                    label="Export JSON",
                    data=json_str,
                    file_name=f"eval_scorecard_{loaded_data['id']}.json",
                    mime="application/json"
                )
            with col_exp2:
                # Convert results to CSV format
                csv_data = {
                    "Metric": [a["agent_name"] for a in agents_outputs] + ["Overall Score"],
                    "Score": [a["score"] for a in agents_outputs] + [score],
                    "Confidence": [a["confidence"] for a in agents_outputs] + [1.0],
                    "Reasoning": [a["reasoning"] for a in agents_outputs] + [grade]
                }
                csv_df = pd.DataFrame(csv_data)
                st.download_button(
                    label="Export CSV",
                    data=csv_df.to_csv(index=False),
                    file_name=f"eval_scorecard_{loaded_data['id']}.csv",
                    mime="text/csv"
                )
        else:
            st.info("Input a question and response on the left, then click 'Evaluate Response' to generate the scorecard.")

    # Bottom detailed breakdown section
    if is_loaded:
        st.divider()
        st.subheader("🔍 Detailed Judge Dimension Reports")
        
        for agent in loaded_data.get("agent_outputs", []):
            with st.expander(f"🤖 {agent['agent_name']} Agent — Score: {agent['score']:.1f}/10.0 (Confidence: {agent['confidence']:.2f})"):
                st.markdown(f"**Reasoning:**\n{agent['reasoning']}")

        # RAG contexts
        contexts = loaded_data.get("retrieved_contexts", [])
        if contexts:
            st.subheader("📚 Retrieved Reference Context (RAG)")
            for idx, ctx in enumerate(contexts):
                with st.expander(f"Context Snippet #{idx+1} — Source: {ctx['source']} (Distance/Similarity: {ctx['score']:.4f})"):
                    st.markdown(ctx["content"])

        # Hallucination Findings
        hallucinated = loaded_data.get("hallucinated_statements", [])
        if hallucinated:
            st.subheader("⚠️ Hallucination Findings (Unsupported Claims)")
            for stmt in hallucinated:
                st.markdown(f"- ❌ *\"{stmt}\"*")

        # Completeness Omissions
        omissions = loaded_data.get("omissions", [])
        if omissions:
            st.subheader("🔍 Completeness Omissions (Missing Details)")
            for om in omissions:
                st.markdown(f"- ⚠️ *\"{om}\"*")

        # Verdict Suggestions
        suggestions = loaded_data.get("suggestions", [])
        if suggestions:
            st.subheader("💡 Suggested Refinements for LLM Output")
            for sug in suggestions:
                st.markdown(f"- ✅ {sug}")


with tab_batch:
    st.header("Batch Response Evaluation")
    st.markdown("""
    Evaluate multiple LLM answers simultaneously by uploading a CSV. 
    The CSV must have the headers: **`question`** and **`response`**. An optional **`reference_answer`** column will enhance accuracy score checks.
    """)

    uploaded_csv = st.file_uploader("Upload Batch CSV file:", type=["csv"])
    batch_use_rag = st.checkbox("Enable RAG context search in batch runs", value=True)

    if uploaded_csv:
        if st.button("Execute Batch Evaluation"):
            with st.spinner("Processing batch rows..."):
                try:
                    files = {"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")}
                    form_data = {"use_rag": str(batch_use_rag)}
                    res = requests.post(f"{BACKEND_URL}/batch-evaluate", files=files, data=form_data)
                    
                    if res.status_code == 200:
                        batch_res = res.json()
                        st.success(f"Batch execution complete! Processed: {batch_res['total_processed']} | Success: {batch_res['success_count']} | Failures: {batch_res['failure_count']}")
                        
                        # Render results table
                        df_res = pd.DataFrame(batch_res["evaluations"])
                        st.dataframe(df_res, use_container_width=True)
                        
                        st.download_button(
                            label="Download Batch Results CSV",
                            data=df_res.to_csv(index=False),
                            file_name="batch_evaluation_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Batch evaluation server error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to execute batch: {e}")

with tab_generate:
    st.header("🤖 RAG QA Generator Agent")
    st.markdown("""
    Ask questions and receive answers grounded in your indexed reference documents, complete with citations.
    """)

    gen_query = st.text_input("Ask a question:", placeholder="e.g. What is the primary engine of a RAG pipeline?", key="gen_query_input")
    
    if st.button("Generate Grounded Answer", disabled=not gen_query, key="gen_btn_run"):
        with st.spinner("Retrieving context and formulating answer..."):
            try:
                res = requests.post(f"{BACKEND_URL}/generate-answer", data={"question": gen_query})
                if res.status_code == 200:
                    gen_data = res.json()
                    answer = gen_data["answer"]
                    citations = gen_data["citations"]

                    # Display Answer
                    st.subheader("Generated Answer:")
                    st.info(answer)

                    # Send to Evaluator shortcut
                    if st.button("Send to Evaluator Tab", key="send_to_eval_shortcut"):
                        st.session_state["eval_question"] = gen_query
                        st.session_state["eval_response"] = answer
                        st.success("Transferred to Single Evaluation tab! Switch tabs above to check scores.")
                        st.rerun()

                    # Display Citations
                    if citations:
                        st.subheader("📚 Source Citations & Reference Contexts:")
                        for idx, cit in enumerate(citations):
                            with st.expander(f"Citation #{idx+1} — Source: {cit['source']} (Distance/Similarity: {cit['score']:.4f})"):
                                st.write(cit["content"])
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")

with tab_analytics:
    st.header("📈 Analytics Dashboard")
    st.markdown("Aggregate metrics and quality trends across all historical evaluations.")
    
    # Load all historical data
    with st.spinner("Loading evaluation history..."):
        all_evals = get_history()
    
    if not all_evals:
        st.info("No evaluation data found. Run some evaluations first!")
    else:
        df_evals = pd.DataFrame(all_evals)
        
        # --- Filters ---
        st.subheader("Filters")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            verdict_filter = st.multiselect(
                "Filter by Verdict", 
                options=["Pass", "Needs Improvement", "Fail"],
                default=["Pass", "Needs Improvement", "Fail"]
            )
        
        # Apply filters
        df_filtered = df_evals[df_evals['final_verdict'].isin(verdict_filter)] if 'final_verdict' in df_evals.columns else df_evals
        
        if df_filtered.empty:
            st.warning("No evaluations match the selected filters.")
        else:
            # --- Top Line Metrics ---
            st.subheader("Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Evaluations", len(df_filtered))
            with col2:
                pass_count = len(df_filtered[df_filtered['final_verdict'] == 'Pass']) if 'final_verdict' in df_filtered.columns else 0
                st.metric("Pass Rate", f"{(pass_count / len(df_filtered) * 100):.1f}%" if len(df_filtered) > 0 else "0%")
            with col3:
                avg_score = df_filtered['overall_score'].mean() if 'overall_score' in df_filtered.columns else 0.0
                st.metric("Avg Overall Score", f"{avg_score:.2f} / 10.0")
            with col4:
                # Hallucination frequency: Percentage of runs with at least 1 hallucination
                if 'hallucinated_statements' in df_filtered.columns:
                    # Convert to list if string
                    def has_hal(x):
                        if isinstance(x, str):
                            try:
                                return len(json.loads(x)) > 0
                            except:
                                return False
                        return isinstance(x, list) and len(x) > 0
                    
                    hal_count = df_filtered['hallucinated_statements'].apply(has_hal).sum()
                    st.metric("Hallucination Rate", f"{(hal_count / len(df_filtered) * 100):.1f}%" if len(df_filtered) > 0 else "0%")
                else:
                    st.metric("Hallucination Rate", "N/A")

            # --- Charts ---
            st.divider()
            st.subheader("Evaluation Visualizations")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Verdict Distribution (Pie Chart)
                if 'final_verdict' in df_filtered.columns:
                    verdict_counts = df_filtered['final_verdict'].value_counts().reset_index()
                    verdict_counts.columns = ['Verdict', 'Count']
                    fig_pie = px.pie(verdict_counts, values='Count', names='Verdict', title='Verdict Distribution', hole=0.4,
                                     color='Verdict', color_discrete_map={'Pass':'#10b981', 'Needs Improvement':'#f59e0b', 'Fail':'#ef4444'})
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                # Average Scores by Dimension
                # We need to extract agent_outputs
                dim_scores = {"Relevance": [], "Accuracy": [], "Completeness": []}
                for _, row in df_filtered.iterrows():
                    if 'agent_outputs' in row and isinstance(row['agent_outputs'], list):
                        for ao in row['agent_outputs']:
                            agent_name = ao.get('agent_name')
                            score = ao.get('score', 0)
                            if agent_name in dim_scores:
                                dim_scores[agent_name].append(score)
                    elif 'agent_outputs' in row and isinstance(row['agent_outputs'], str):
                        try:
                            agent_outs = json.loads(row['agent_outputs'])
                            for ao in agent_outs:
                                agent_name = ao.get('agent_name')
                                score = ao.get('score', 0)
                                if agent_name in dim_scores:
                                    dim_scores[agent_name].append(score)
                        except:
                            pass
                
                avg_dims = {k: (sum(v)/len(v) if v else 0) for k, v in dim_scores.items()}
                
                fig_bar = px.bar(
                    x=list(avg_dims.keys()), 
                    y=list(avg_dims.values()),
                    title="Average Dimension Scores",
                    labels={'x': 'Dimension', 'y': 'Average Score'},
                    color=list(avg_dims.keys()),
                    color_discrete_map={'Relevance': '#3b82f6', 'Accuracy': '#6366f1', 'Completeness': '#8b5cf6'}
                )
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                fig_bar.update_yaxes(range=[0, 10])
                st.plotly_chart(fig_bar, use_container_width=True)

            # Trend over time
            if 'created_at' in df_filtered.columns:
                st.subheader("Quality Trends Over Time")
                # Ensure created_at is datetime
                # Handle ISO strings
                try:
                    df_trend = df_filtered.copy()
                    df_trend['created_at'] = pd.to_datetime(df_trend['created_at'])
                    df_trend = df_trend.sort_values('created_at')
                    fig_line = px.line(df_trend, x='created_at', y='overall_score', title='Overall Score Trend', markers=True)
                    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                    st.plotly_chart(fig_line, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not parse dates for trend chart: {e}")
            
            # --- Export ---
            st.divider()
            st.subheader("Generate & Download PDF Report")
            st.markdown("Compile the currently filtered data into a comprehensive PDF report.")
            if st.button("Generate PDF", key="btn_gen_pdf"):
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_bytes = generate_pdf_report(df_filtered, report_title="Filtered Evaluation Analytics Report")
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name="evaluation_report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate PDF: {e}")
