import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.app.config import settings

# Gracefully handle sentence-transformers loading in case of offline/network issues
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorStoreManager:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.index_path = os.path.join(settings.VECTOR_DB_PATH, "faiss.index")
        self.metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.pkl")
        
        self.model = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        # Load models and indexes
        self._initialize_model()
        self._load_index()

    def _initialize_model(self):
        """Initializes the embedding model safely."""
        if not EMBEDDINGS_AVAILABLE:
            print("[Warning] sentence-transformers is not installed. Using mock embeddings.")
            return

        try:
            print(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load embedding model: {e}. Falling back to mock embeddings.")
            self.model = None

    def _load_index(self):
        """Loads FAISS index from disk or initializes a new one."""
        if not FAISS_AVAILABLE:
            print("[Warning] FAISS is not installed. Indexing operations will be mocked.")
            return

        # Determine dimensions (all-MiniLM-L6-v2 is 384)
        dimension = 384
        if self.model is not None:
            try:
                dimension = self.model.get_sentence_embedding_dimension()
            except AttributeError:
                pass

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded existing FAISS index with {len(self.metadata)} documents.")
            except Exception as e:
                print(f"[Error] Failed to load FAISS index: {e}. Reinitializing index.")
                self.index = faiss.IndexFlatL2(dimension)
                self.metadata = []
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings using sentence-transformers or mock vectors."""
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"[Error] Embedding generation failed: {e}. Using mock embeddings.")

        # Mock embedding logic (random normalized vectors for durability)
        dimension = 384
        embeddings = []
        for text in texts:
            # Seed based on text hash for deterministic mock behavior
            np.random.seed(abs(hash(text)) % (2**32))
            vec = np.random.randn(dimension)
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        """Indexes text chunks and their metadata into FAISS and persists changes."""
        if not chunks:
            return

        embeddings = self.get_embeddings(chunks)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embeddings)
            
            for chunk, meta in zip(chunks, metadatas):
                self.metadata.append({
                    "text": chunk,
                    **meta
                })
            self._save_index()
        else:
            # Mock persistence in metadata list
            for chunk, meta in zip(chunks, metadatas):
                self.metadata.append({
                    "text": chunk,
                    **meta
                })
            print(f"[Mock] Added {len(chunks)} documents to memory database.")

    def _save_index(self):
        """Persists the FAISS index and metadata pickle file to disk."""
        if not FAISS_AVAILABLE or self.index is None:
            return

        try:
            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print("FAISS index and metadata saved to disk.")
        except Exception as e:
            print(f"[Error] Failed to save FAISS index: {e}")

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Searches the top-k similar documents and returns them with their distances."""
        if not self.metadata:
            return []

        # If k is greater than metadata size, cap it
        k = min(k, len(self.metadata))

        query_vector = self.get_embeddings([query])

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(query_vector, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata) and idx >= 0:
                    results.append((self.metadata[idx], float(dist)))
            return results
        else:
            # Mock search using simple word-overlap ratio since FAISS/embeddings are missing
            print("[Mock] Performing word overlap query matching.")
            query_words = set(query.lower().split())
            scored = []
            for doc in self.metadata:
                doc_words = set(doc["text"].lower().split())
                overlap = len(query_words.intersection(doc_words))
                score = overlap / max(len(query_words), 1)
                # Convert score to an L2 distance equivalent (higher score = lower distance)
                scored.append((doc, float(1.0 - score)))
            
            scored.sort(key=lambda x: x[1])
            return scored[:k]
