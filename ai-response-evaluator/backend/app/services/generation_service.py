import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from backend.app.config import settings
from backend.app.rag.pipeline import rag_pipeline

class GenerationService:
    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes the LLM generator or returns None for mock fallback."""
        if settings.LLM_PROVIDER == "openai":
            api_key = settings.OPENAI_API_KEY
            if not api_key or api_key == "mock-key" or api_key == "mock-api-key-or-empty":
                return None
            try:
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL_NAME,
                    api_key=api_key,
                    base_url=settings.OPENAI_API_BASE,
                    temperature=0.3
                )
            except Exception as e:
                print(f"[GenerationService] Failed to init ChatOpenAI: {e}")
                return None
        elif settings.LLM_PROVIDER == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    base_url=settings.OLLAMA_API_BASE,
                    model=settings.OLLAMA_MODEL_NAME,
                    temperature=0.3
                )
            except Exception as e:
                print(f"[GenerationService] Failed to init ChatOllama: {e}")
                return None
        return None

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Retrieves context from FAISS and generates an answer citing the references."""
        # 1. Retrieve relevant contexts
        contexts = rag_pipeline.retrieve_context(question, k=3)
        
        if not contexts:
            return {
                "answer": "No relevant reference documents were found in the knowledge library. Please index documents first.",
                "citations": []
            }

        # 2. Formulate answer
        if self.llm is None:
            # Fallback to mock generation using context snippets
            answer = self._mock_generate_answer(question, contexts)
        else:
            try:
                # Merge retrieved text blocks for context injection
                merged_context = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['content']}" for ctx in contexts])

                system_prompt = (
                    "You are a helpful, expert AI Question-Answering Agent. Your goal is to answer the user's question "
                    "accurately, using ONLY the provided source contexts. If the answer cannot be found in the context, "
                    "say that you cannot find the answer. Always cite your sources by referencing their Source name "
                    "within your response where appropriate (e.g. [squad_bootstrap] or [filename.pdf])."
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", f"Contexts:\n{merged_context}\n\nQuestion: {question}")
                ])
                
                chain = prompt | self.llm
                response = chain.invoke({})
                answer = response.content.strip()
            except Exception as e:
                print(f"[GenerationService] LLM generation failed: {e}. Using mock fallback.")
                answer = self._mock_generate_answer(question, contexts)

        return {
            "answer": answer,
            "citations": contexts
        }

    def _mock_generate_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Helper mock compiler summarizing context sentences to answer the query."""
        best_context = contexts[0]
        source_tag = best_context["source"]
        content = best_context["content"]
        
        # Build a response incorporating the best match context text
        ans = (
            f"Based on the reference source [{source_tag}], the knowledge base indicates that: "
            f"\"{content[:300]}...\". This matches details relating to your query about '{question}'."
        )
        return ans

# Instantiate singleton
generation_service = GenerationService()
