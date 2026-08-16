import os
import re
import json
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from backend.app.config import settings

class BaseJudgeAgent:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes the configured LLM provider or returns None for mock execution."""
        # Check config and API keys
        if settings.LLM_PROVIDER == "openai":
            api_key = settings.OPENAI_API_KEY
            if not api_key or api_key == "mock-key" or api_key == "mock-api-key-or-empty":
                print(f"[{self.name}] No valid OpenAI API key provided. Using local heuristics mock fallback.")
                return None
            try:
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL_NAME,
                    api_key=api_key,
                    base_url=settings.OPENAI_API_BASE,
                    temperature=0.0
                )
            except Exception as e:
                print(f"[{self.name}] Failed to init ChatOpenAI: {e}. Using mock fallback.")
                return None
        
        elif settings.LLM_PROVIDER == "ollama":
            # Lazy import to avoid loading issues if dependency isn't fully ready
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    base_url=settings.OLLAMA_API_BASE,
                    model=settings.OLLAMA_MODEL_NAME,
                    temperature=0.0
                )
            except Exception as e:
                print(f"[{self.name}] Failed to init ChatOllama: {e}. Using mock fallback.")
                return None

        return None

    def execute_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Calls the active LLM using the structured prompts, with robust JSON parsing."""
        if self.llm is None:
            # Fallback to mock scoring if no LLM is configured or loaded
            return self.mock_evaluate(user_prompt)

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\nIMPORTANT: Return ONLY a valid JSON object. Do not include markdown tags or surrounding text."),
                ("human", user_prompt)
            ])
            chain = prompt | self.llm
            response = chain.invoke({})
            content = response.content.strip()

            # Robust JSON cleaning and parsing
            return self._parse_json_response(content)
        except Exception as e:
            print(f"[{self.name}] LLM invocation failed: {e}. Falling back to mock scoring.")
            return self.mock_evaluate(user_prompt)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Strips markdown and parses JSON outputs robustly."""
        # Strip markdown json codeblocks if present
        cleaned = text
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                cleaned = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                cleaned = match.group(1)

        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to regex pull score, reasoning, and confidence if format is broken
            score_match = re.search(r'"score"\s*:\s*([\d\.]+)', cleaned)
            confidence_match = re.search(r'"confidence"\s*:\s*([\d\.]+)', cleaned)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', cleaned)

            score = float(score_match.group(1)) if score_match else 5.0
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reasoning = reasoning_match.group(1) if reasoning_match else f"Fallback parsed from broken JSON content: {text[:100]}"
            
            return {
                "score": score,
                "reasoning": reasoning,
                "confidence": confidence
            }

    def mock_evaluate(self, user_prompt: str) -> Dict[str, Any]:
        """Fallback evaluation using deterministic string analysis heuristics."""
        raise NotImplementedError("Each judge agent must implement its own mock_evaluate logic.")
