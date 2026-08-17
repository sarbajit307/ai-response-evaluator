# AI Response Quality Evaluator Agent

An advanced, production-ready multi-agent evaluation platform built in Python. This system analyzes, evaluates, and grades Large Language Model responses across multiple criteria using specialized judging agents and Retrieval-Augmented Generation (RAG) context verification.

---

## Key Features
* **Multi-Agent Evaluation Panel**: Runs specialized judging agents in parallel (Relevance Judge, Accuracy Judge, Hallucination/Groundedness Judge, Completeness Judge).
* **RAG Grounding**: Local vector base indexing using `FAISS` and sentence embeddings (`all-MiniLM-L6-v2`) to verify generated claims against source documents.
* **Verdict Synthesis**: Automatically calculates a final weighted scorecard, assigns an overall grade (A, B, C, F), and provides actionable suggestions for output refinements.
* **REST API**: Built with `FastAPI` to execute evaluation runs, manage reference libraries, and load historical data.
* **Interactive Dashboard**: Sleek `Streamlit` client rendering scorecards, interactive Plotly radar charts, details of judge reasonings, and RAG context segments.
* **Batch Processing**: Bulk evaluation via CSV upload.
* **Offline Resilience**: Automatically falls back to deterministic rule-based evaluation heuristics if no LLM API keys are provided, allowing instant out-of-the-box local runs.

---

## Folder Structure
```text
ai-response-evaluator/
├── docs/                        # Project documentation
│   ├── research/                # LLM & RAG evaluation literature
│   └── architecture/            # System designs and API specifications
├── backend/
│   └── app/
│       ├── main.py              # FastAPI application server entry point
│       ├── config.py            # Pydantic environment configuration
│       ├── database/            # SQLAlchemy schemas & sessions
│       ├── schemas/             # Pydantic request/response validation
│       ├── api/                 # REST routes
│       ├── rag/                 # Chunking, Embedding, & FAISS vector search
│       ├── agents/              # Core judge agent implementations
│       └── services/            # Evaluation orchestration pipeline
├── frontend/
│   └── app.py                   # Streamlit dashboard interface
├── tests/                       # Unit and API integration tests
├── .env.example                 # Config template
├── Dockerfile                   # Single-image runner config
├── docker-compose.yml           # Compose configuration (Backend + Frontend)
└── requirements.txt             # Project dependencies
```

---

## Installation & Setup

### Prerequisites
* Python 3.10 or 3.11
* Internet connection (on first startup to fetch embedding models, or runs on mocks offline)

### Step 1: Clone and Configure Environment
1. Extract or clone this directory.
2. Copy the config template to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Set your LLM provider options inside `.env`:
   ```text
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your-actual-api-key
   ```
   *(Leave key as mock to run the system with offline rule-based heuristic judges).*

### Step 2: Install Dependencies
Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the Application

#### Option A: Running Locally (Fastest)
1. Start the FastAPI backend:
   ```bash
   python -m backend.app.main
   ```
   *(FastAPI runs on `http://127.0.0.1:8000`)*

2. Open a separate terminal, activate virtual environment, and start Streamlit:
   ```bash
   streamlit run frontend/app.py
   ```
   *(Streamlit opens on `http://127.0.0.1:8501`)*

#### Option B: Docker Compose (Production Setup)
Start both containers concurrently:
```bash
docker-compose up --build
```
* Backend API: `http://localhost:8000`
* Frontend Dashboard: `http://localhost:8501`

---

## Running Tests
Run tests locally via `pytest` to assert API endpoints and agent scoring validations:
```bash
python -m pytest
```

---

## API Endpoints
* **`GET /health`**: Health diagnostics and model status.
* **`POST /evaluate`**: Run single question evaluation (supports prompt, response, reference, and PDF/TXT upload).
* **`POST /batch-evaluate`**: Evaluate batch dataset uploads via CSV.
* **`POST /upload-reference`**: Index external references into the FAISS index.
* **`GET /results`**: Pull past evaluations.

---

## Future Improvements
* **Advanced RAG Reranking**: Integrate Cohere or cross-encoder rerankers to improve top-k context alignment.
* **Active LLM Guardrails**: Incorporate Llama Guard or NeMo Guardrails directly into the ingestion phase.
* **Dynamic Weighting**: Allow users to slide weights for Relevance, Accuracy, Completeness, and Groundedness metrics directly from the UI.
