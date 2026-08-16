from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Requests
class EvaluationRequest(BaseModel):
    question: str = Field(..., description="The user query or prompt.")
    response: str = Field(..., description="The candidate model response.")
    reference_answer: Optional[str] = Field(None, description="Optional ground-truth reference answer.")
    use_rag: bool = Field(True, description="Whether to fetch context from vector store.")

class ReferenceUploadRequest(BaseModel):
    text: str = Field(..., description="Plain text to index in knowledge base.")
    source_name: str = Field("api_upload", description="Source metadata identifier.")

# Responses
class AgentOutputSchema(BaseModel):
    agent_name: str
    score: float
    confidence: float
    reasoning: str

    class Config:
        from_attributes = True

class RetrievedContextSchema(BaseModel):
    content: str
    source: Optional[str]
    score: Optional[float]

    class Config:
        from_attributes = True

class EvaluationResponse(BaseModel):
    id: int
    question: str
    response: str
    reference_answer: Optional[str] = None
    overall_score: float
    final_grade: str
    final_verdict: Optional[str] = None
    synthesis: Optional[str] = None
    suggestions: Optional[List[str]] = []
    hallucinated_statements: Optional[List[str]] = []
    omissions: Optional[List[str]] = []
    created_at: datetime
    agent_outputs: List[AgentOutputSchema] = []
    retrieved_contexts: List[RetrievedContextSchema] = []

    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    status: str
    database: str
    vector_store: str
    llm_provider: str
    llm_model: str
