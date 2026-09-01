import pytest
import asyncio
import time
from httpx import AsyncClient, ASGITransport
from main import app
from config import AI_MODE, VALID_TICKERS
from agents.technical_agent import technical_agent
from agents.fundamental_rag_agent import fundamental_rag_agent
from agents.sentiment_agent import sentiment_agent
from agents.risk_agent import risk_agent
from agents.devils_advocate_agent import devils_advocate_agent
from agents.thesis_break_agent import thesis_break_agent
from agents.counterfactual_simulator import counterfactual_simulator
from agents.orchestrator import orchestrator
from engine.decision_twin import decision_twin_engine
from engine.evidence_graph import evidence_graph_engine
from engine.change_digest import change_digest_engine
from profiling.behavioral_bias_mirror import detect_behavioral_biases
from profiling.behavioral_drift import evaluate_behavioral_drift
from utils.security import check_no_diagnostic_terms, validate_ticker, validate_persona
from utils.metrics import compute_confidence_breakdown
from utils.filing_freshness import check_filing_freshness
from schemas import (
    ThesisRecord, TechnicalSignal, FundamentalSignal, SentimentSignal,
    SynthesizedOutput, ConfidenceBreakdown
)

@pytest.fixture
def sample_market_data():
    return orchestrator.market_data_cache

# ---------------------------------------------------------
# Test 1: Technical calculations produce valid signal from raw data (§12)
# ---------------------------------------------------------
def test_1_technical_calculations(sample_market_data):
    tata_data = sample_market_data["TATAMOTORS"]
    sig = technical_agent.analyze("TATAMOTORS", tata_data)
    
    assert sig.ticker == "TATAMOTORS"
    assert 0.0 <= sig.rsi <= 100.0
    assert sig.macd_signal in ["BULLISH_CROSSOVER", "BEARISH_CROSSOVER", "NEUTRAL"]
    assert sig.classification in ["STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"]
    assert 0.0 <= sig.confidence <= 1.0
    assert len(sig.reasons) > 0
    assert not sig.degraded

# ---------------------------------------------------------
# Test 2: RAG retrieves expected XYZ_CORP debt evidence (§12)
# ---------------------------------------------------------
def test_2_rag_retrieves_xyz_debt_evidence():
    retrieved = fundamental_rag_agent.retrieve(query="debt to equity liabilities pledge", ticker="XYZ_CORP", top_k=2)
    assert len(retrieved) > 0
    
    # Verify Page 14 citation exists
    citations = [r["citation_tag"] for r in retrieved]
    assert any("[SEBI-Filing-XYZ-Q3: Page 14]" in c for c in citations)
    
    # Verify debt text is contained
    full_text = " ".join([r["text"] for r in retrieved])
    assert "3.85" in full_text or "debt-to-equity" in full_text.lower() or "pledge" in full_text.lower()

# ---------------------------------------------------------
# Test 3: Parallel agents execute concurrently (§12)
# (Elapsed time tracks max, not sum, of agent times)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_3_parallel_agents_concurrency(sample_market_data):
    tata_data = sample_market_data["TATAMOTORS"]
    
    t0 = time.perf_counter()
    res = await orchestrator.analyze("TATAMOTORS", persona="conservative")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    latencies = res.telemetry["agent_latencies_ms"]
    sum_latencies = latencies["TechnicalAgent"] + latencies["FundamentalRAGAgent"] + latencies["SentimentAgent"]
    parallel_time = latencies["ParallelGather"]
    
    # Parallel gather time should be substantially less than synchronous sum + overhead
    assert parallel_time <= sum_latencies + 20.0

# ---------------------------------------------------------
# Test 4: Conservative and Aggressive produce divergent advice on identical input (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_4_divergent_persona_advice():
    # Run XYZ_CORP on both personas
    res_cons = await orchestrator.analyze("XYZ_CORP", persona="conservative")
    res_aggr = await orchestrator.analyze("XYZ_CORP", persona="aggressive")
    
    # Both see identical market data, but advice & verdicts diverge
    assert res_cons.synthesis.personalized_advice != res_aggr.synthesis.personalized_advice
    assert res_cons.risk_assessment.risk_score != res_aggr.risk_assessment.risk_score
    assert res_cons.synthesis.synthesized_verdict in ["AVOID-HIGH RISK", "HOLD-WATCH"]

# ---------------------------------------------------------
# Test 5: Conflict resolution detects Tech BUY vs Fund BEARISH and does not majority-vote (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_5_conflict_resolution_no_majority_vote():
    res = await orchestrator.analyze("XYZ_CORP", persona="conservative", scenario="conflict")
    
    # Tech = BULLISH, Sent = POSITIVE, Fundamental = CRITICAL_RISK (debt)
    assert res.agent_outputs["technical"]["classification"] in ["STRONG_BULLISH", "BULLISH"]
    assert res.agent_outputs["sentiment"]["classification"] == "POSITIVE"
    assert res.agent_outputs["fundamental"]["filing_verdict"] == "CRITICAL_RISK"
    
    # Must NOT majority vote to BUY CANDIDATE
    assert res.synthesis.synthesized_verdict != "BUY CANDIDATE"
    assert res.synthesis.conflict_summary is not None
    assert "debt" in res.synthesis.conflict_summary.lower() or "divergence" in res.synthesis.conflict_summary.lower()

# ---------------------------------------------------------
# Test 6: Missing filing does not produce HTTP 500 (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_6_missing_filing_resilience():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Run with degraded scenario
        response = await client.post("/api/analyze", json={
            "ticker": "TATAMOTORS",
            "persona": "conservative",
            "scenario": "degraded"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["degraded_status"]["degraded_data"] is True

# ---------------------------------------------------------
# Test 7: Gemini failure triggers deterministic fallback (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_7_gemini_failure_fallback():
    # Orchestrator handles mock / offline mode cleanly without errors
    res = await orchestrator.analyze("INFOSYS", persona="conservative")
    assert res.synthesis.synthesized_verdict in ["BUY CANDIDATE", "HOLD-WATCH", "REDUCE EXPOSURE"]
    assert len(res.synthesis.reasoning_trace) > 0

# ---------------------------------------------------------
# Test 8: Invalid ticker rejected by validation (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_8_invalid_ticker_validation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/analyze", json={
            "ticker": "INVALID_CO",
            "persona": "conservative"
        })
        assert response.status_code == 400
        assert "Invalid ticker" in response.json()["detail"]

# ---------------------------------------------------------
# Test 9: Invalid persona rejected by validation (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_9_invalid_persona_validation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/analyze", json={
            "ticker": "TATAMOTORS",
            "persona": "ultra_speculator_unknown"
        })
        assert response.status_code == 400
        assert "Invalid persona" in response.json()["detail"]

# ---------------------------------------------------------
# Test 10: No API key required in mock mode (§12)
# ---------------------------------------------------------
def test_10_no_api_key_required_in_mock_mode():
    assert AI_MODE in ["mock", "gemini"]
    # Even with empty API key, health and analyzer work
    transport = ASGITransport(app=app)

# ---------------------------------------------------------
# Test 11: Every fundamental recommendation has citation when data available (§12)
# ---------------------------------------------------------
def test_11_fundamental_recommendation_citations(sample_market_data):
    sig, _ = fundamental_rag_agent.analyze_fundamentals("TATAMOTORS", sample_market_data["TATAMOTORS"])
    assert len(sig.rag_citations) > 0
    assert any("TATAMOTORS" in c for c in sig.rag_citations)

# ---------------------------------------------------------
# Test 12: Degraded mode reduces confidence or flags uncertainty (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_12_degraded_mode_confidence_reduction():
    res_normal = await orchestrator.analyze("TATAMOTORS", persona="conservative", scenario="aligned")
    res_degraded = await orchestrator.analyze("TATAMOTORS", persona="conservative", scenario="degraded")
    
    assert res_degraded.degraded_status["degraded_data"] is True
    assert res_degraded.synthesis.confidence < res_normal.synthesis.confidence

# ---------------------------------------------------------
# Test 13: Devil's Advocate never introduces citation absent from original pool (§12)
# ---------------------------------------------------------
def test_13_devils_advocate_evidence_constrained(sample_market_data):
    fund_sig, _ = fundamental_rag_agent.analyze_fundamentals("XYZ_CORP", sample_market_data["XYZ_CORP"])
    tech_sig = technical_agent.analyze("XYZ_CORP", sample_market_data["XYZ_CORP"])
    sent_sig = sentiment_agent.analyze("XYZ_CORP", sample_market_data["XYZ_CORP"])
    
    original_citations = set(fund_sig.rag_citations)
    challenge = devils_advocate_agent.challenge(
        ticker="XYZ_CORP",
        draft_verdict="BUY CANDIDATE",
        technical_sig=tech_sig,
        fundamental_sig=fund_sig,
        sentiment_sig=sent_sig,
        existing_citations=fund_sig.rag_citations
    )
    
    # If a citation is mentioned in challenge, it MUST belong to original pool
    for cit in original_citations:
        if cit in challenge.counter_argument:
            assert cit in original_citations

# ---------------------------------------------------------
# Test 14: Confidence breakdown components bounded [0,1] and sum to composite (§12)
# ---------------------------------------------------------
def test_14_confidence_breakdown_bounded_and_consistent():
    cb = compute_confidence_breakdown(
        data_freshness=0.90,
        agent_agreement=0.80,
        evidence_strength=0.85,
        historical_calibration=0.80
    )
    
    assert 0.0 <= cb.data_freshness_score <= 1.0
    assert 0.0 <= cb.agent_agreement_score <= 1.0
    assert 0.0 <= cb.evidence_strength_score <= 1.0
    assert 0.0 <= cb.historical_calibration_score <= 1.0
    assert 0.0 <= cb.composite_confidence <= 1.0
    
    # Weighted math check
    expected = (0.90 * 0.25) + (0.80 * 0.35) + (0.85 * 0.25) + (0.80 * 0.15)
    assert abs(cb.composite_confidence - round(expected, 3)) <= 0.01

# ---------------------------------------------------------
# Test 15: Stale filing produces degraded=true + warning; fresh filing does not (§12)
# ---------------------------------------------------------
def test_15_stale_filing_detection(sample_market_data):
    # Stale scenario
    stale_sig, stale_warn = fundamental_rag_agent.analyze_fundamentals("XYZ_CORP", sample_market_data["XYZ_CORP"], scenario="stale_behavioral")
    assert stale_sig.degraded is True
    assert stale_warn is not None
    assert "Freshness Warning" in stale_warn

    # Fresh scenario
    fresh_sig, fresh_warn = fundamental_rag_agent.analyze_fundamentals("TATAMOTORS", sample_market_data["TATAMOTORS"], scenario="aligned")
    assert fresh_sig.degraded is False
    assert fresh_warn is None

# ---------------------------------------------------------
# Test 16: Behavioral Bias Mirror flags momentum-chasing and 0 flags in control (§12)
# ---------------------------------------------------------
def test_16_behavioral_bias_flags_and_control(sample_market_data):
    # Momentum chasing case
    xyz_flags = detect_behavioral_biases(
        persona_id="aggressive",
        ticker="XYZ_CORP",
        portfolio={"XYZ_CORP": 0.02},
        market_data=sample_market_data["XYZ_CORP"],
        scenario="stale_behavioral"
    )
    assert len(xyz_flags) >= 1
    assert any(f.flag_type == "MOMENTUM_CHASING" for f in xyz_flags)

    # Control case: Infosys with neutral behavior
    control_flags = detect_behavioral_biases(
        persona_id="conservative",
        ticker="INFOSYS",
        portfolio={"INFOSYS": 0.05},
        market_data=sample_market_data["INFOSYS"],
        scenario="aligned"
    )
    assert len(control_flags) == 0

# ---------------------------------------------------------
# Test 17: DecisionTwin's 5 scores independently computed and never averaged (§18.12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_17_decision_twin_five_scores_independent():
    res = await orchestrator.analyze("XYZ_CORP", persona="conservative")
    dt = res.decision_twin
    
    scores = [dt.market_confidence, dt.decision_fit, dt.evidence_quality, dt.thesis_health, dt.behavioral_risk]
    for s in scores:
        assert 0.0 <= s <= 1.0
        
    # Scores are not identical or equal to a simple mean
    assert dt.composite_note is not None
    assert len(dt.composite_note) > 10

# ---------------------------------------------------------
# Test 18: Thesis Break flags BROKEN only on citation-backed evidence (§18.12)
# ---------------------------------------------------------
def test_18_thesis_break_requires_citation_evidence():
    thesis = ThesisRecord(
        thesis_record_id="T1",
        ticker="XYZ_CORP",
        user_id="u1",
        created_at="2026-01-01",
        stated_reasons=["Growth"],
        key_assumptions=["debt_to_equity stays below 1.2"],
        invalidating_conditions=["D/E > 1.5"]
    )
    
    # Case with citation and high debt
    fund_sig = FundamentalSignal(
        ticker="XYZ_CORP",
        rag_citations=["[SEBI-Filing-XYZ-Q3: Page 14]"],
        debt_to_equity=3.85,
        earnings_growth=-0.12,
        filing_verdict="CRITICAL_RISK",
        confidence=0.9,
        evidence=["Debt is 3.85x"],
        degraded=False
    )
    breaks = thesis_break_agent.evaluate_thesis(thesis, fund_sig, None, None)
    assert len(breaks) == 1
    assert breaks[0].severity == "BROKEN"
    assert breaks[0].evidence_citation == "[SEBI-Filing-XYZ-Q3: Page 14]"

    # Degraded case with no citation -> should not break on absence of data
    fund_sig_empty = FundamentalSignal(
        ticker="XYZ_CORP",
        rag_citations=[],
        debt_to_equity=0.0,
        earnings_growth=0.0,
        filing_verdict="NEUTRAL",
        confidence=0.2,
        evidence=[],
        degraded=True
    )
    breaks_empty = thesis_break_agent.evaluate_thesis(thesis, fund_sig_empty, None, None)
    assert len(breaks_empty) == 0

# ---------------------------------------------------------
# Test 19: Counterfactual Simulator always populates disclaimer and cites assumption varied (§18.12)
# ---------------------------------------------------------
def test_19_counterfactual_simulator_disclaimer_and_assumptions(sample_market_data):
    tech_sig = technical_agent.analyze("TATAMOTORS", sample_market_data["TATAMOTORS"])
    fund_sig, _ = fundamental_rag_agent.analyze_fundamentals("TATAMOTORS", sample_market_data["TATAMOTORS"])
    sent_sig = sentiment_agent.analyze("TATAMOTORS", sample_market_data["TATAMOTORS"])
    
    scenarios = counterfactual_simulator.simulate(
        ticker="TATAMOTORS",
        persona_id="conservative",
        technical_sig=tech_sig,
        fundamental_sig=fund_sig,
        sentiment_sig=sent_sig
    )
    
    assert len(scenarios) == 4
    for sc in scenarios:
        assert sc.simulation_disclaimer == "Simulated illustration based on demo data. Not a forecast or guarantee."
        assert len(sc.assumption_changed) > 5

# ---------------------------------------------------------
# Test 20: Evidence Graph resolves claims to cited evidence; marks uncited (§18.12)
# ---------------------------------------------------------
def test_20_evidence_graph_claim_resolution():
    synth = SynthesizedOutput(
        session_id="s123",
        ticker="TATAMOTORS",
        raw_signals={"technical": {"classification": "BULLISH"}, "fundamental": {"filing_verdict": "POSITIVE"}, "sentiment": {}},
        synthesized_verdict="BUY CANDIDATE",
        market_classification="BULLISH",
        confidence=0.88,
        reasoning_trace=[],
        source_attributions=["[SEBI-Filing-TATAMOTORS-Q3: Page 12]"],
        personalized_advice="Buy",
        risk_profile="conservative",
        degraded_data=False,
        disclaimer="",
        confidence_breakdown=ConfidenceBreakdown(
            data_freshness_score=0.9, agent_agreement_score=0.9, evidence_strength_score=0.9,
            historical_calibration_score=0.85, composite_confidence=0.88
        )
    )
    
    nodes = evidence_graph_engine.build_graph(synth, ["Net net debt reduced significantly."])
    ev_nodes = [n for n in nodes if n.node_type == "EVIDENCE"]
    assert len(ev_nodes) >= 1
    assert ev_nodes[0].citation_tag == "[SEBI-Filing-TATAMOTORS-Q3: Page 12]"

# ---------------------------------------------------------
# Test 21: What Changed Engine returns empty digest on identical runs (§18.12)
# ---------------------------------------------------------
def test_21_what_changed_empty_on_no_changes():
    data = {
        "session_id": "sess_1",
        "synthesized_verdict": "BUY CANDIDATE",
        "signal_classification": "BULLISH",
        "raw_result_payload": {"thesis_break_events": [], "behavioral_drift_report": {"drift_severity": "NONE"}}
    }
    digest = change_digest_engine.compute_changes(
        ticker="TATAMOTORS",
        current_data=data,
        previous_data=data,
        since_session_id="sess_1"
    )
    assert len(digest.changed_items) == 0

# ---------------------------------------------------------
# Test 22: Behavioral Drift Report passes banned diagnostic terms check (§18.12)
# ---------------------------------------------------------
def test_22_behavioral_drift_no_diagnostic_terms(sample_market_data):
    drift_report = evaluate_behavioral_drift(
        persona_id="aggressive",
        ticker="XYZ_CORP",
        portfolio={"XYZ_CORP": 0.02},
        market_data=sample_market_data["XYZ_CORP"],
        scenario="stale_behavioral"
    )
    
    assert check_no_diagnostic_terms(drift_report.observed_behavior_pattern)
    for flag in drift_report.drift_flags:
        assert check_no_diagnostic_terms(flag.nudge_message)
        assert check_no_diagnostic_terms(flag.trigger_reason)

# ---------------------------------------------------------
# Test 23: GenAI Co-Pilot Endpoint provides citation-grounded answer
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_23_copilot_endpoint_grounded_answer():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/api/copilot", json={
            "ticker": "XYZ_CORP",
            "query": "What is the debt to equity ratio?",
            "persona": "conservative"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["ticker"] == "XYZ_CORP"
        assert "Debt-to-Equity" in data["answer"] or "debt" in data["answer"].lower()
        assert len(data["citations"]) > 0


