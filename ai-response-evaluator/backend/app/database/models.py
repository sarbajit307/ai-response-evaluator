from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.database.session import Base

class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    response = Column(String, nullable=False)
    reference_answer = Column(String, nullable=True)
    overall_score = Column(Float, nullable=False)
    final_grade = Column(String, nullable=False)
    final_verdict = Column(String, nullable=True) # Pass/Needs Improvement/Fail
    suggestions = Column(String, nullable=True) # JSON list
    hallucinated_statements = Column(String, nullable=True) # JSON list
    omissions = Column(String, nullable=True) # JSON list
    synthesis = Column(String, nullable=True) # Consolidated reasoning summary
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    agent_outputs = relationship("AgentOutput", back_populates="evaluation", cascade="all, delete-orphan")
    retrieved_contexts = relationship("RetrievedContext", back_populates="evaluation", cascade="all, delete-orphan")


class AgentOutput(Base):
    __tablename__ = "agent_outputs"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False)
    agent_name = Column(String, nullable=False) # e.g. relevance, accuracy, hallucination, completeness
    score = Column(Float, nullable=False) # 0-10
    confidence = Column(Float, nullable=False) # 0-1
    reasoning = Column(String, nullable=False)

    # Relationships
    evaluation = relationship("Evaluation", back_populates="agent_outputs")


class RetrievedContext(Base):
    __tablename__ = "retrieved_contexts"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False)
    content = Column(String, nullable=False)
    source = Column(String, nullable=True) # e.g. "squad", "truthful_qa", "uploaded_doc.pdf"
    score = Column(Float, nullable=True) # Cosine distance / similarity score

    # Relationships
    evaluation = relationship("Evaluation", back_populates="retrieved_contexts")
