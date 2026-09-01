import pytest
import inspect
from agents.fundamental_rag_agent import fundamental_rag_agent, FundamentalRAGAgent
from agents.orchestrator import orchestrator
from schemas import FundamentalSignal

# ---------------------------------------------------------
# Test 1: XYZ debt value has complete evidence provenance
# ---------------------------------------------------------
def test_1_xyz_debt_value_provenance():
    sig, _ = fundamental_rag_agent.analyze_fundamentals("XYZ_CORP", {})
    assert sig.debt_to_equity == 3.85
    assert sig.filing_verdict == "CRITICAL_RISK"
    assert sig.evidence_provenance is not None
    assert "debt_to_equity" in sig.evidence_provenance
    
    prov = sig.evidence_provenance["debt_to_equity"]
    assert prov["value"] == 3.85
    assert prov["citation"] == "[SEBI-Filing-XYZ-Q3: Page 14]"
    assert prov["page"] == 14

# ---------------------------------------------------------
# Test 2: Citation comes directly from retrieved evidence
# ---------------------------------------------------------
def test_2_citation_comes_from_retrieved_evidence():
    retrieved = fundamental_rag_agent.retrieve("debt liabilities covenants", "XYZ_CORP", top_k=2)
    sig, _ = fundamental_rag_agent.analyze_fundamentals("XYZ_CORP", {})
    
    retrieved_citations = {c["citation_tag"] for c in retrieved}
    for cit in sig.rag_citations:
        assert cit in retrieved_citations

# ---------------------------------------------------------
# Test 3: Missing/Unknown filing produces UNKNOWN / NULL
# ---------------------------------------------------------
def test_3_missing_filing_produces_unknown_and_null():
    sig, warning = fundamental_rag_agent.analyze_fundamentals("NON_EXISTENT_CORP", {})
    assert sig.debt_to_equity is None
    assert sig.earnings_growth is None
    assert sig.filing_verdict in ["UNKNOWN", "INSUFFICIENT_EVIDENCE"]
    assert sig.data_available is False
    assert sig.confidence == 0.0

# ---------------------------------------------------------
# Test 4: Missing evidence never becomes 0.0
# ---------------------------------------------------------
def test_4_missing_evidence_never_becomes_zero():
    sig_degraded, _ = fundamental_rag_agent.analyze_fundamentals("TATAMOTORS", {}, scenario="degraded")
    assert sig_degraded.debt_to_equity is None
    assert sig_degraded.debt_to_equity != 0.0
    assert sig_degraded.debt_to_equity != 0

# ---------------------------------------------------------
# Test 5: Source code check: No ticker-specific financial assignments exist
# ---------------------------------------------------------
def test_5_no_ticker_specific_financial_assignments_in_source():
    source = inspect.getsource(fundamental_rag_agent.analyze_fundamentals)
    assert 'if ticker == "XYZ_CORP":' not in source
    assert 'if ticker == "TATAMOTORS":' not in source
    assert 'if ticker == "INFOSYS":' not in source
    assert 'debt_to_equity = 3.85' not in source
    assert 'debt_to_equity = 0.62' not in source
    assert 'debt_to_equity = 0.08' not in source

# ---------------------------------------------------------
# Test 6: Scenario B still detects the XYZ debt warning
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_6_scenario_b_detects_xyz_debt_warning():
    res = await orchestrator.analyze("XYZ_CORP", persona="conservative", scenario="conflict")
    assert res.agent_outputs["fundamental"]["debt_to_equity"] == 3.85
    assert res.agent_outputs["fundamental"]["filing_verdict"] == "CRITICAL_RISK"
    assert res.synthesis.synthesized_verdict in ["HOLD-WATCH", "AVOID-HIGH RISK"]
    assert "[SEBI-Filing-XYZ-Q3: Page 14]" in res.citations

# ---------------------------------------------------------
# Test 7: Scenario C (Degraded mode) still works properly
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_7_scenario_c_degraded_mode():
    res = await orchestrator.analyze("INFOSYS", persona="conservative", scenario="degraded")
    assert res.degraded_status["degraded_data"] is True
    assert res.agent_outputs["fundamental"]["debt_to_equity"] is None
    assert res.agent_outputs["fundamental"]["degraded"] is True

# ---------------------------------------------------------
# Test 8: TATAMOTORS and INFOSYS values derived from evidence
# ---------------------------------------------------------
def test_8_tata_and_infosys_provenance():
    sig_tata, _ = fundamental_rag_agent.analyze_fundamentals("TATAMOTORS", {})
    assert sig_tata.debt_to_equity == 0.62
    assert sig_tata.filing_verdict == "POSITIVE"
    assert sig_tata.evidence_provenance["debt_to_equity"]["citation"] == "[SEBI-Filing-TATAMOTORS-Q3: Page 12]"
    assert sig_tata.evidence_provenance["debt_to_equity"]["page"] == 12

    sig_infy, _ = fundamental_rag_agent.analyze_fundamentals("INFOSYS", {})
    assert sig_infy.debt_to_equity == 0.08
    assert sig_infy.filing_verdict == "POSITIVE"
    assert sig_infy.evidence_provenance["debt_to_equity"]["citation"] == "[SEBI-Filing-INFOSYS-Q3: Page 8]"
    assert sig_infy.evidence_provenance["debt_to_equity"]["page"] == 8
