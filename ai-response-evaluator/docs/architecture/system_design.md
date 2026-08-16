# System Design and Architecture Documentation

This document describes the architectural blueprint, data flows, components, API designs, and pipeline logic of the **AI Response Quality Evaluator Agent**.

---

## 1. System Architecture & Component Diagram

The system is designed following **Clean Architecture** principles to separate core business logic (evaluators, agents, and pipelines) from delivery mechanisms (FastAPI REST API, Streamlit Frontend) and data access layers (SQLite, FAISS Vector Index).

### Component Diagram

```mermaid
graph TD
    subgraph Frontend [Presentation Layer]
        UI[Streamlit Dashboard App]
    end

    subgraph API [API Layer]
        FastAPI[FastAPI Server]
        Routes[API Routes /evaluate, /batch-evaluate, /upload-reference]
        Schemas[Pydantic Request/Response Schemas]
    end

    subgraph Service [Core Business Logic]
        EvalService[Evaluation Service]
        Scoring[Scoring Module]
        
        subgraph Multi-Agent System
            VA[Verdict Agent]
            RJ[Relevance Judge]
            AJ[Accuracy Judge]
            HJ[Hallucination Judge]
            CJ[Completeness Judge]
        end
    end

    subgraph RAG [RAG / Retrieval Layer]
        RAGPipeline[RAG Pipeline]
        DocLoader[Document & Dataset Loader]
        FAISS[FAISS Vector Store]
        Embedder[SentenceTransformer Embeddings]
    end

    subgraph Data [Persistence Layer]
        SQLite[(SQLite DB via SQLAlchemy)]
    end

    %% Interactions
    UI -->|HTTP / REST| Route[Routes]
    Route --> Schemas
    Route -->|Invokes| EvalService
    
    EvalService --> RAGPipeline
    RAGPipeline --> DocLoader
    RAGPipeline --> FAISS
    FAISS --> Embedder
    
    EvalService -->|Triggers| VA
    VA --> RJ
    VA --> AJ
    VA --> HJ
    VA --> CJ
    
    EvalService --> Scoring
    EvalService -->|Persists Data| SQLite
    
    UI -->|Queries Results| Route
```

---

## 2. Sequence Diagram

This diagram shows the end-to-end execution flow of a single evaluation query.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Streamlit
    participant API as FastAPI Server
    participant EvalSvc as Evaluation Service
    participant RAG as RAG Pipeline
    participant Judge as Judge Agents (Multi-Agent Panel)
    participant Verdict as Verdict Agent
    participant DB as SQLite / SQLAlchemy

    User->>API: POST /evaluate (Question, Response, optional files/refs)
    Note over API: Validate Request Schemas
    
    API->>EvalSvc: evaluate_response(...)
    
    alt Optional Source Document uploaded or reference query triggered
        EvalSvc->>RAG: retrieve_context(Question)
        RAG->>RAG: Generate query embedding
        RAG->>RAG: FAISS similarity search
        RAG-->>EvalSvc: Top-k Context segments
    else No reference document
        Note over EvalSvc: Use user-provided reference text
    end

    par Parallel Judge Execution
        EvalSvc->>Judge: Run Relevance Agent (Prompt, Response)
        Judge-->>EvalSvc: Relevance Score + Reasoning
    and
        EvalSvc->>Judge: Run Accuracy Agent (Prompt, Response, Reference)
        Judge-->>EvalSvc: Accuracy Score + Reasoning
    and
        EvalSvc->>Judge: Run Hallucination Agent (Prompt, Response, Context)
        Judge-->>EvalSvc: Hallucination Score + Reasoning
    and
        EvalSvc->>Judge: Run Completeness Agent (Prompt, Response)
        Judge-->>EvalSvc: Completeness Score + Reasoning
    end

    EvalSvc->>Verdict: compile_verdict(all scores & justifications)
    Note over Verdict: Calculate weighted score & grade<br/>Formulate suggestions
    Verdict-->>EvalSvc: Final Scorecard

    EvalSvc->>DB: Save Evaluation, Contexts, & Agent Scorecards
    DB-->>EvalSvc: Save success

    EvalSvc-->>API: Return scorecard JSON
    API-->>User: Render Dashboard (Charts, Scorecard, History)
```

---

## 3. Database Design

We use SQLite for relational persistence. The schema details are below:

```mermaid
erDiagram
    EVALUATION {
        int id PK
        string question
        string response
        string reference_answer
        float overall_score
        string final_grade
        string suggestions
        datetime created_at
    }

    AGENT_OUTPUT {
        int id PK
        int evaluation_id FK
        string agent_name
        float score
        float confidence
        string reasoning
    }

    RETRIEVED_CONTEXT {
        int id PK
        int evaluation_id FK
        string content
        string source
        float score
    }

    EVALUATION ||--o{ AGENT_OUTPUT : triggers
    EVALUATION ||--o{ RETRIEVED_CONTEXT : references
```

### Table Definitions
1. **`Evaluation`**: Main evaluation execution record.
2. **`AgentOutput`**: Stores the raw score, reasoning text, and confidence calculated by each individual specialized LLM agent.
3. **`RetrievedContext`**: Holds chunks of relevant text extracted from RAG datasets or uploaded files.

---

## 4. Evaluation and Scoring Pipeline

The evaluation system scores responses on a 0-10 scale for multiple dimensions.

### Scoring Dimensions
* **Relevance**: Evaluates if the response is directly addressing the prompt.
* **Accuracy**: Evaluates alignment with the reference ground truth.
* **Groundedness (Faithfulness)**: Evaluates if response claims are fully backed by retrieved context.
* **Completeness**: Evaluates if all elements of the question are answered.

### Scoring Engine Logic
The overall score is a weighted sum of the individual dimensions. The default weights are:
$$\text{Overall Score} = 0.25 \times \text{Relevance} + 0.25 \times \text{Accuracy} + 0.30 \times \text{Groundedness} + 0.20 \times \text{Completeness}$$

### Grades
- **Grade A (Excellent)**: $\ge 9.0$
- **Grade B (Good)**: $\ge 7.5$ and $< 9.0$
- **Grade C (Fair)**: $\ge 5.5$ and $< 7.5$
- **Grade F (Fail)**: $< 5.5$

---

## 5. RAG Pipeline

```mermaid
graph LR
    Doc[TXT / PDF / SQuAD Dataset] --> Chunker[Recursive Chunker]
    Chunker --> Embedder[Sentence-Transformers]
    Embedder --> Indexer[FAISS Vector DB]
    
    Query[User Question] --> QueryEmbedder[Sentence-Transformers]
    QueryEmbedder --> Search[FAISS L2 Search]
    Search --> Context[Top-k Context builder]
```

* **Chunking**: Chunks documents into 500-character segments with a 50-character overlap.
* **Indexing**: Chunks are processed via BAAI or MiniLM embedding models and saved locally in a FAISS index.
* **Retrieval**: At query time, the top 3 (k=3) contexts are fetched to populate the grounding prompt.

---

## 6. API Design

### `GET /health`
Verifies server health and model status.

### `POST /upload-reference`
Allows uploading reference TXT/PDF files or text to insert into the FAISS index.
- **Request (Multipart)**:
  - `file`: Uploaded file (optional)
  - `text`: Raw text (optional)
  - `source_name`: String identifier

### `POST /evaluate`
Executes single response evaluation.
- **Request (JSON / Multipart)**:
  - `question`: Query string
  - `response`: Generated response
  - `reference_answer`: Ground truth (optional)
- **Response**: Full scorecard including scores, individual agent logs, retrieved contexts, and verdict.

### `POST /batch-evaluate`
Accepts a CSV with `question`, `response`, and optional `reference_answer` columns, running evaluations on each row.

### `GET /results`
Retrieves past records from the SQLite database.
