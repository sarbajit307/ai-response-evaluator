import pytest
from backend.app.agents.relevance import RelevanceJudge
from backend.app.agents.accuracy import AccuracyJudge
from backend.app.agents.completeness import CompletenessJudge
import statistics

def test_agent_scoring_consistency():
    q = "Explain photosynthesis."
    a = "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
    ref = "Plants convert light energy into chemical energy."
    
    relevance_scores = []
    accuracy_scores = []
    completeness_scores = []
    
    rj = RelevanceJudge()
    aj = AccuracyJudge()
    cj = CompletenessJudge()
    
    for _ in range(5):
        r_out = rj.evaluate(q, a)
        a_out = aj.evaluate(q, a, reference_answer=ref)
        c_out = cj.evaluate(q, a)
        
        relevance_scores.append(r_out["score"])
        accuracy_scores.append(a_out["score"])
        completeness_scores.append(c_out["score"])
        
    # Standard deviation should be 0 since our logic is fully deterministic heuristic based currently
    assert statistics.stdev(relevance_scores) == 0.0
    assert statistics.stdev(accuracy_scores) == 0.0
    assert statistics.stdev(completeness_scores) == 0.0
