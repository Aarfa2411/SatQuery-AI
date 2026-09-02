from app.schemas import QueryRequest
from app.services.agent import react_agent
from app.services.exporter import report_exporter


def test_deterministic_signature_reproducibility():
    checksums = ["sha256_hash_1", "sha256_hash_2"]
    query = "What changed between these dates?"
    model_versions = {"change_detection": "v1.0"}
    metrics = {"pct_change": 12.4}
    final_ans = "Built-up area increased by 12.4%."

    # Run 1
    h1 = react_agent._compute_deterministic_signature(checksums, query, model_versions, metrics, final_ans)
    # Run 2 (different time, same inputs)
    h2 = react_agent._compute_deterministic_signature(checksums, query, model_versions, metrics, final_ans)
    assert h1 == h2
    assert len(h1) == 64

def test_pdf_report_and_geojson_generation():
    req = QueryRequest(session_id="export_sess_test", query="Identify water bodies.")
    resp = react_agent.run_query(req)

    geojson_path = report_exporter.export_geojson("export_sess_test", resp)
    pdf_path = report_exporter.export_pdf("export_sess_test", resp)

    assert geojson_path.exists()
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
