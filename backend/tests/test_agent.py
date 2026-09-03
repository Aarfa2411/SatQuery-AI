from app.schemas import QueryRequest
from app.services.agent import react_agent
from app.services.output_sanity_checker import output_sanity_checker


def test_confidence_fusion_boundaries():
    # Test extreme warnings to ensure confidence is always bounded [0.05, 1.0]
    class MockToolOut:
        def __init__(self, c, w):
            self.tool = "test"
            self.model_version = "v1"
            self.confidence = c
            self.warnings = ["warn"] * w
            self.metrics = {}

    # 15 warnings
    outs = [MockToolOut(0.8, 15)]
    final_c, breakdown, warnings = react_agent._fuse_confidences(outs)
    assert 0.05 <= final_c <= 1.0

def test_geovalidator_bbox_clamping():
    raw_bboxes = [
        {"label": "OutOfBounds", "box": [-0.2, 0.1, 1.5, 0.9], "score": 1.2},
        {"label": "Valid", "box": [0.1, 0.1, 0.8, 0.8], "score": 0.95},
        {"label": "Inverted", "box": [0.8, 0.8, 0.2, 0.2], "score": 0.5}
    ]
    cleaned, warnings = output_sanity_checker.sanitize_bboxes(raw_bboxes)
    assert len(cleaned) == 2  # Inverted should be dropped
    assert cleaned[0].box == [0.0, 0.1, 1.0, 0.9]
    assert cleaned[0].score == 1.0
    assert len(warnings) > 0

def test_react_agent_execution_trace():
    req = QueryRequest(session_id="test_sess_001", query="Describe the satellite image.")
    resp = react_agent.run_query(req)
    assert resp.session_id == "test_sess_001"
    assert len(resp.trace) >= 1
    assert resp.trace[0].why_this_tool != ""
    assert 0.05 <= resp.confidence <= 1.0
    assert resp.execution_mode in ("real_model", "heuristic_fallback", "hybrid")
