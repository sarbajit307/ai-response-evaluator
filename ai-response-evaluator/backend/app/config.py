import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./evaluator.db"

    # LLM Settings
    LLM_PROVIDER: str = "openai" # 'openai' or 'ollama' or 'mock'
    OPENAI_API_KEY: Optional[str] = "mock-key"
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    # Ollama Settings
    OLLAMA_API_BASE: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "llama3"

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH: str = "./faiss_index"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()

# Ensure directories exist
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
