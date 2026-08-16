import os
from typing import List, Dict, Any
from backend.app.rag.loader import DocumentLoader
from backend.app.rag.vector_store import VectorStoreManager

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStoreManager()

    def bootstrap_knowledge_base(self):
        """Indexes default SQuAD & TruthfulQA examples if index is empty."""
        # If we already have indexed documents, skip bootstrapping
        if len(self.vector_store.metadata) > 0:
            print("Knowledge base already bootstrapped. Skipping.")
            return

        print("Bootstrapping reference knowledge base with SQuAD & TruthfulQA datasets...")
        datasets = DocumentLoader.get_bootstrap_datasets()
        chunks = []
        metadatas = []

        for item in datasets:
            context_text = item["context"]
            # Split context text into small chunks
            text_chunks = DocumentLoader.chunk_text(context_text, chunk_size=150, chunk_overlap=20)
            for chunk in text_chunks:
                chunks.append(chunk)
                metadatas.append({
                    "question": item["question"],
                    "source": item["source"]
                })

        self.vector_store.add_documents(chunks, metadatas)
        print("Bootstrapping complete.")

    def index_text(self, text: str, source_name: str) -> int:
        """Indexes raw text string. Returns number of indexed chunks."""
        if not text.strip():
            return 0

        cleaned = DocumentLoader.clean_text(text)
        chunks = DocumentLoader.chunk_text(cleaned, chunk_size=200, chunk_overlap=30)
        
        metadatas = [{"source": source_name} for _ in chunks]
        self.vector_store.add_documents(chunks, metadatas)
        
        return len(chunks)

    def index_file(self, file_path: str, source_name: str) -> int:
        """Indexes a PDF or TXT file. Returns number of indexed chunks."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            text = DocumentLoader.load_pdf(file_path)
        elif ext == ".txt":
            text = DocumentLoader.load_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Only PDF and TXT are supported.")

        return self.index_text(text, source_name)

    def retrieve_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-k context snippets relevant to query."""
        search_results = self.vector_store.similarity_search(query, k=k)
        retrieved = []
        for doc, score in search_results:
            retrieved.append({
                "content": doc["text"],
                "source": doc.get("source", "unknown"),
                "score": score
            })
        return retrieved

# Singleton instance for simple dependency injection
rag_pipeline = RAGPipeline()
