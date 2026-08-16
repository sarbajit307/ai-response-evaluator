import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.database.session import init_db
from backend.app.rag.pipeline import rag_pipeline
from backend.app.api.routes import router

# Initialize database schema
print("Initializing database...")
init_db()

# Bootstrap vector knowledge base
try:
    print("Bootstrapping RAG vector indexes...")
    rag_pipeline.bootstrap_knowledge_base()
except Exception as e:
    print(f"[Warning] Failed to bootstrap vector store on startup: {e}")

# Build FastAPI Application
app = FastAPI(
    title="AI Response Quality Evaluator Agent API",
    description="Multi-agent grading and RAG verification backend for Large Language Models.",
    version="1.0.0"
)

# Setup CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="")

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
