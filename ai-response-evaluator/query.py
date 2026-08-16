import httpx
import sys
import json

# Setup local FastAPI server endpoint
API_URL = "http://127.0.0.1:8000/evaluate"

def run_fast_evaluation():
    # Setup test payload
    payload = {
        "question": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "reference_answer": "Paris is the capital and largest city of France.",
        "use_rag": "false"
    }

    try:
        # Request evaluation from the active, running server
        response = httpx.post(API_URL, data=payload, timeout=5.0)
        
        if response.status_code == 200:
            result = response.json()
            print("--- EVALUATION SCORECARD ---")
            print(f"Overall Score: {result['overall_score']:.2f} / 10.0")
            print(f"Verdict: {result.get('final_verdict', 'N/A')} (Grade: {result['final_grade']})")
            
            synthesis = result.get("synthesis", "")
            if synthesis:
                print(f"Consolidated Reasoning: {synthesis}")
            
            print("\nBreakdown:")
            for agent in result['agent_outputs']:
                print(f" - {agent['agent_name']}: {agent['score']:.1f}/10 ({agent['reasoning']})")
                
            hallucinations = result.get("hallucinated_statements", [])
            if hallucinations:
                print("\nFlagged Hallucinations:")
                for stmt in hallucinations:
                    print(f" [❌] {stmt}")
                    
            omissions = result.get("omissions", [])
            if omissions:
                print("\nCompleteness Omissions:")
                for om in omissions:
                    print(f" [🔍] {om}")
        else:
            print(f"Server returned error status: {response.status_code}")
    except Exception as e:
        print(f"Failed to connect to the server: {e}")
        print("Please ensure the backend server is running (py -m backend.app.main).")

if __name__ == "__main__":
    run_fast_evaluation()
