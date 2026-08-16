import io
import json

def test_health_endpoint(client):
    """Verifies that the /health check reports status successfully."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data
    assert data["llm_provider"] == "mock"


def test_evaluate_endpoint(client):
    """Verifies single response evaluation route completes validation and scores correctly."""
    form_data = {
        "question": "What are the benefits of modular code?",
        "response": "Modular code is clean, reusable, easy to debug, and satisfies SOLID principles.",
        "reference_answer": "Modular code provides high reusability, isolatable testing, and clean interfaces.",
        "use_rag": "false"
    }
    
    response = client.post("/evaluate", data=form_data)
    assert response.status_code == 200
    data = response.json()
    
    # Assert return structure
    assert data["question"] == form_data["question"]
    assert data["response"] == form_data["response"]
    assert "overall_score" in data
    assert "final_grade" in data
    assert len(data["agent_outputs"]) == 4


def test_upload_reference_endpoint(client):
    """Verifies text indexing into the mock knowledge base is processed successfully."""
    form_data = {
        "text": "FAISS is optimized for rapid similarity search of dense vectors.",
        "source_name": "test_upload"
    }
    
    response = client.post("/upload-reference", data=form_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] > 0


def test_results_endpoint(client):
    """Verifies that past evaluations saved inside the database can be retrieved."""
    # Ensure there's at least one record by running an evaluation first
    form_data = {
        "question": "What is the capital of France?",
        "response": "Paris.",
        "use_rag": "false"
    }
    client.post("/evaluate", data=form_data)

    response = client.get("/results")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["question"] == "What is the capital of France?"


def test_batch_evaluate_endpoint(client):
    """Verifies batch evaluation upload parsing and processing."""
    csv_content = (
        "question,response,reference_answer\n"
        "What is FAISS?,FAISS is a vector search database.,Meta developed FAISS for similarity queries.\n"
        "Explain RAG.,RAG connects context indices to generation.,RAG joins retriever and generator modules.\n"
    )
    
    file_payload = {
        "file": ("batch_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    }
    
    response = client.post("/batch-evaluate", files=file_payload, data={"use_rag": "false"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["total_processed"] == 2
    assert data["success_count"] == 2
