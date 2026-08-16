import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_single_evaluation_workflow():
    data = {
        "question": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "use_rag": "false"
    }
    response = client.post("/evaluate", data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert "overall_score" in res_json
    assert "final_verdict" in res_json
    
def test_batch_evaluation_workflow():
    csv_content = b"question,response\nWhat is 2+2?,4\nWhat is the sun?,A star."
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/batch-evaluate", files=files, data={"use_rag": "false"})
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["total_processed"] == 2
    assert len(res_json["evaluations"]) == 2
    assert "final_verdict" in res_json["evaluations"][0]

def test_results_endpoint():
    response = client.get("/results")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
