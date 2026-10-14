from omnibrain_core import OmniBrainOrchestrator, SupervisorAgent


def test_supervisor_routes_chart_query_to_vision():
    result = SupervisorAgent().route("Explain this chart")
    assert "vision" in result["routes"]


def test_supervisor_routes_revenue_query_to_sql_and_retrieval():
    result = SupervisorAgent().route("Compare monthly revenue")
    assert "sql" in result["routes"]
    assert "retrieval" in result["routes"]


def test_orchestrator_handles_empty_documents():
    result = OmniBrainOrchestrator().process_query("Summarize the report", documents=[])
    assert isinstance(result["answer"], str)
    assert "citations" in result
    assert result["confidence"] == 0.0
