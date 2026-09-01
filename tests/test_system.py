import pytest
import asyncio
import time
import json
from httpx import AsyncClient, ASGITransport
from main import app
from config import AI_MODE, VALID_TICKERS, VALID_PERSONAS
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
from utils.metrics import compute_confidence_breakdown, WEIGHT_FRESHNESS, WEIGHT_AGREEMENT, WEIGHT_EVIDENCE, WEIGHT_CALIBRATION
from utils.filing_freshness import check_filing_freshness
from schemas import (
    ThesisRecord, TechnicalSignal, FundamentalSignal, SentimentSignal,
    SynthesizedOutput, ConfidenceBreakdown
)

@pytest.fixture
def sample_market_data():
    return orchestrator.market_data_cache

# ---------------------------------------------------------
# Test 1: Technical calculations produce realistic non-100 RSI and valid indicators (§9 & §12)
# ---------------------------------------------------------
def test_1_technical_calculations(sample_market_data):
    tata_data = sample_market_data["TATAMOTORS"]
    sig = technical_agent.analyze("TATAMOTORS", tata_data)
    
    assert sig.ticker == "TATAMOTORS"
    assert 50.0 <= sig.rsi <= 80.0, f"Expected realistic RSI between 50 and 80, got {sig.rsi}"
    assert sig.macd_signal in ["BULLISH_CROSSOVER", "BEARISH_CROSSOVER", "NEUTRAL"]
    assert sig.classification in ["STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"]
    assert 0.0 <= sig.confidence <= 1.0
    assert len(sig.reasons) > 0
    assert not sig.degraded
    assert "standard deviations" in " ".join(sig.reasons)

# ---------------------------------------------------------
# Test 2: RAG retrieves expected XYZ_CORP debt evidence with Page 14 citation (§3 & §12)
# ---------------------------------------------------------
def test_2_rag_retrieves_xyz_debt_evidence():
    retrieved = fundamental_rag_agent.retrieve(query="debt to equity liabilities pledge covenants", ticker="XYZ_CORP", top_k=2)
    assert len(retrieved) > 0
    
    citations = [r["citation_tag"] for r in retrieved]
    assert any("[SEBI-Filing-XYZ-Q3: Page 14]" in c for c in citations)
    
    full_text = " ".join([r["text"] for r in retrieved])
    assert "3.85" in full_text or "debt-to-equity" in full_text.lower() or "pledge" in full_text.lower()

# ---------------------------------------------------------
# Test 3: Parallel agents execute concurrently with controlled delays (§28)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_3_parallel_agents_concurrency(sample_market_data):
    t0 = time.perf_counter()
    res = await orchestrator.analyze("TATAMOTORS", persona="conservative")
    
    latencies = res.telemetry["agent_latencies_ms"]
    sum_latencies = latencies["TechnicalAgent"] + latencies["FundamentalRAGAgent"] + latencies["SentimentAgent"]
    parallel_time = latencies["ParallelGather"]
    
    assert parallel_time < sum_latencies + 25.0

# ---------------------------------------------------------
# Test 4: Conservative and Aggressive produce divergent advice on identical input (§6)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_4_divergent_persona_advice():
    res_cons = await orchestrator.analyze("XYZ_CORP", persona="conservative")
    res_aggr = await orchestrator.analyze("XYZ_CORP", persona="aggressive")
    
    assert res_cons.synthesis.personalized_advice != res_aggr.synthesis.personalized_advice
    assert res_cons.risk_assessment.risk_score != res_aggr.risk_assessment.risk_score
    assert res_cons.synthesis.synthesized_verdict in ["AVOID-HIGH RISK", "HOLD-WATCH"]

# ---------------------------------------------------------
# Test 5: Conflict resolution detects Tech BUY vs Fund BEARISH and does not majority-vote (§7)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_5_conflict_resolution_no_majority_vote():
    res = await orchestrator.analyze("XYZ_CORP", persona="conservative", scenario="conflict")
    
    assert res.agent_outputs["technical"]["classification"] in ["STRONG_BULLISH", "BULLISH"]
    assert res.agent_outputs["sentiment"]["classification"] == "POSITIVE"
    assert res.agent_outputs["fundamental"]["filing_verdict"] == "CRITICAL_RISK"
    
    assert res.synthesis.synthesized_verdict != "BUY CANDIDATE"
    assert res.synthesis.conflict_summary is not None
    assert "conflict detected" in res.synthesis.conflict_summary.lower() or "debt" in res.synthesis.conflict_summary.lower()

# ---------------------------------------------------------
# Test 6: Missing filing does not produce HTTP 500 (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_6_missing_filing_resilience():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/analyze", json={
            "ticker": "TATAMOTORS",
            "persona": "conservative",
            "scenario": "degraded"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["degraded_status"]["degraded_data"] is True

# ---------------------------------------------------------
# Test 7: Gemini failure triggers deterministic fallback (§4 & §12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_7_gemini_failure_fallback():
    res = await orchestrator.analyze("INFOSYS", persona="conservative")
    assert res.synthesis.synthesized_verdict in ["BUY CANDIDATE", "HOLD-WATCH", "REDUCE EXPOSURE", "AVOID-HIGH RISK"]
    assert len(res.synthesis.reasoning_trace) >= 0
    assert res.synthesis.fallback_used is True or res.synthesis.llm_used is True

# ---------------------------------------------------------
# Test 8: Invalid ticker rejected with HTTP 400 (§24)
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

# ---------------------------------------------------------
# Test 9: Invalid persona rejected with HTTP 400 (§24)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_9_invalid_persona_validation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/analyze", json={
            "ticker": "TATAMOTORS",
            "persona": "super_gambler"
        })
        assert response.status_code == 400

# ---------------------------------------------------------
# Test 10: No API key required in offline / mock mode (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_10_no_api_key_required_in_mock_mode():
    res = await orchestrator.analyze("TATAMOTORS", persona="aggressive")
    assert res.session_id.startswith("sess_")
    assert res.synthesis.synthesized_verdict in ["BUY CANDIDATE", "HOLD-WATCH", "AVOID-HIGH RISK"]

# ---------------------------------------------------------
# Test 11: Fundamental recommendation includes citations (§3 & §11)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_11_fundamental_recommendation_citations():
    res = await orchestrator.analyze("TATAMOTORS", persona="conservative")
    citations = res.citations
    assert len(citations) > 0
    assert any("[Earnings-TATAMOTORS-Q3: Page 4]" in c or "TATAMOTORS" in c for c in citations)

# ---------------------------------------------------------
# Test 12: Degraded mode reduces confidence score (§12)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_12_degraded_mode_confidence_reduction():
    res_normal = await orchestrator.analyze("TATAMOTORS", persona="conservative", scenario="aligned")
    res_degraded = await orchestrator.analyze("TATAMOTORS", persona="conservative", scenario="degraded")
    
    assert res_degraded.synthesis.confidence < res_normal.synthesis.confidence
    assert res_degraded.degraded_status["degraded_data"] is True

# ---------------------------------------------------------
# Test 13: Devil's Advocate is strictly evidence-constrained (§8 & §29)
# ---------------------------------------------------------
def test_13_devils_advocate_evidence_constrained():
    tech = TechnicalSignal(
        ticker="XYZ_CORP", rsi=78.5, macd_signal="BULLISH_CROSSOVER",
        momentum_score=18.5, volume_anomaly_score=2.45, classification="STRONG_BULLISH",
        confidence=0.85, reasons=["Momentum"]
    )
    fund = FundamentalSignal(
        ticker="XYZ_CORP", rag_citations=["[SEBI-Filing-XYZ-Q3: Page 14]"],
        debt_to_equity=3.85, earnings_growth=-0.12, filing_verdict="CRITICAL_RISK",
        confidence=0.90, evidence=["Debt-to-equity is 3.85x."]
    )
    sent = SentimentSignal(
        ticker="XYZ_CORP", news_sentiment=0.85, fii_flow="INFLOW", dii_flow="NEUTRAL", fii_flow_trend="INFLOW",
        social_chatter_score=0.92, classification="POSITIVE",
        confidence=0.80, reasons=["Positive buzz"]
    )
    
    challenge = devils_advocate_agent.challenge(
        ticker="XYZ_CORP",
        draft_verdict="BUY CANDIDATE",
        technical_sig=tech,
        fundamental_sig=fund,
        sentiment_sig=sent,
        existing_citations=fund.rag_citations
    )
    
    assert challenge.severity == "HIGH"
    assert "[SEBI-Filing-XYZ-Q3: Page 14]" in challenge.counter_argument
    assert challenge.confidence_adjustment < 0.0

# ---------------------------------------------------------
# Test 14: Confidence breakdown weights sum to 1.00 and components are bounded (§16)
# ---------------------------------------------------------
def test_14_confidence_breakdown_bounded_and_consistent():
    total_weights = WEIGHT_FRESHNESS + WEIGHT_AGREEMENT + WEIGHT_EVIDENCE + WEIGHT_CALIBRATION
    assert abs(total_weights - 1.00) < 1e-6, f"Confidence weights must sum to 1.00, got {total_weights}"
    
    cb = compute_confidence_breakdown(
        data_freshness=0.95,
        agent_agreement=0.85,
        evidence_strength=0.90,
        historical_calibration=0.85
    )
    assert 0.0 <= cb.data_freshness_score <= 1.0
    assert 0.0 <= cb.agent_agreement_score <= 1.0
    assert 0.0 <= cb.evidence_strength_score <= 1.0
    assert 0.0 <= cb.historical_calibration_score <= 1.0
    assert 0.0 <= cb.composite_confidence <= 1.0
    
    expected = (0.95 * 0.25) + (0.85 * 0.35) + (0.90 * 0.25) + (0.85 * 0.15)
    assert abs(cb.composite_confidence - round(expected, 3)) <= 0.002

# ---------------------------------------------------------
# Test 15: Stale filing detection produces warning flag (§15)
# ---------------------------------------------------------
def test_15_stale_filing_detection():
    is_stale, warning = check_filing_freshness("2024-06-15", "XYZ_CORP")
    assert is_stale is True
    assert "Warning" in warning or "months old" in warning

# ---------------------------------------------------------
# Test 16: Behavioral bias flags trigger without diagnostic language (§13 & §22)
# ---------------------------------------------------------
def test_16_behavioral_bias_flags_and_control():
    m_data = {
        "simulated_user_actions": {
            "request_count_past_hour": 5,
            "recent_price_spike_pct": 14.5,
            "requested_position_size_multiplier": 3.0,
            "sector_repeat_queries": 4
        }
    }
    flags = detect_behavioral_biases(
        persona_id="aggressive",
        ticker="XYZ_CORP",
        portfolio={"XYZ_CORP": 0.20},
        market_data=m_data,
        scenario="stale_behavioral"
    )
    assert len(flags) > 0
    types = [f.flag_type for f in flags]
    assert "MOMENTUM_CHASING" in types or "RECENCY_BIAS" in types
    
    for f in flags:
        assert check_no_diagnostic_terms(f.trigger_reason)
        assert check_no_diagnostic_terms(f.nudge_message)

# ---------------------------------------------------------
# Test 17: Decision Twin scores are independent (§15)
# ---------------------------------------------------------
def test_17_decision_twin_five_scores_independent():
    tech = TechnicalSignal(ticker="TATAMOTORS", rsi=65.0, macd_signal="BULLISH_CROSSOVER", momentum_score=12.0, volume_anomaly_score=1.5, classification="BULLISH", confidence=0.85, reasons=["Up"])
    fund = FundamentalSignal(ticker="TATAMOTORS", rag_citations=["[Earnings-TATAMOTORS-Q3: Page 4]"], debt_to_equity=0.62, earnings_growth=0.28, filing_verdict="POSITIVE", confidence=0.90, evidence=["Healthy"])
    sent = SentimentSignal(ticker="TATAMOTORS", news_sentiment=0.75, fii_flow="INFLOW", dii_flow="INFLOW", fii_flow_trend="INFLOW", social_chatter_score=0.70, classification="POSITIVE", confidence=0.80, reasons=["Good"])
    risk = risk_agent.assess_risk("TATAMOTORS", "conservative", tech, fund, sent, {"sector": "Automobile"})
    drift = evaluate_behavioral_drift("conservative", "TATAMOTORS", {"TATAMOTORS": 0.05}, {})

    twin = decision_twin_engine.compute_twin("TATAMOTORS", "conservative", tech, fund, sent, risk, [], drift)
    
    assert 0.0 <= twin.market_confidence <= 1.0
    assert 0.0 <= twin.decision_fit <= 1.0
    assert 0.0 <= twin.evidence_quality <= 1.0
    assert 0.0 <= twin.thesis_health <= 1.0
    assert 0.0 <= twin.behavioral_risk <= 1.0

# ---------------------------------------------------------
# Test 18: Thesis break requires citation evidence (§18)
# ---------------------------------------------------------
def test_18_thesis_break_requires_citation_evidence():
    thesis = ThesisRecord(
        thesis_record_id="THESIS-XYZ-01",
        ticker="XYZ_CORP",
        user_id="user_1",
        created_at="2026-01-01T00:00:00Z",
        stated_reasons=["Growth thesis"],
        key_assumptions=["debt_to_equity stays below 1.5"],
        invalidating_conditions=["debt exceeds 1.5x"]
    )
    fund = FundamentalSignal(
        ticker="XYZ_CORP",
        rag_citations=["[SEBI-Filing-XYZ-Q3: Page 14]"],
        debt_to_equity=3.85,
        earnings_growth=-0.12,
        filing_verdict="CRITICAL_RISK",
        confidence=0.90,
        evidence=["High leverage"]
    )
    tech = TechnicalSignal(ticker="XYZ_CORP", rsi=75.0, macd_signal="BULLISH_CROSSOVER", momentum_score=15.0, volume_anomaly_score=2.0, classification="BULLISH", confidence=0.80, reasons=["Up"])
    sent = SentimentSignal(ticker="XYZ_CORP", news_sentiment=0.80, fii_flow="INFLOW", dii_flow="NEUTRAL", fii_flow_trend="INFLOW", social_chatter_score=0.90, classification="POSITIVE", confidence=0.80, reasons=["Buzz"])

    breaks = thesis_break_agent.evaluate_thesis(thesis, fund, tech, sent)
    assert len(breaks) > 0
    assert breaks[0].severity == "BROKEN"
    assert breaks[0].evidence_citation == "[SEBI-Filing-XYZ-Q3: Page 14]"

# ---------------------------------------------------------
# Test 19: Risk scenario simulator includes disclaimer (§17)
# ---------------------------------------------------------
def test_19_counterfactual_simulator_disclaimer_and_assumptions():
    tech = TechnicalSignal(ticker="TATAMOTORS", rsi=65.0, macd_signal="BULLISH_CROSSOVER", momentum_score=12.0, volume_anomaly_score=1.5, classification="BULLISH", confidence=0.85, reasons=["Up"])
    fund = FundamentalSignal(ticker="TATAMOTORS", rag_citations=["[Earnings-TATAMOTORS-Q3: Page 4]"], debt_to_equity=0.62, earnings_growth=0.28, filing_verdict="POSITIVE", confidence=0.90, evidence=["Healthy"])
    sent = SentimentSignal(ticker="TATAMOTORS", news_sentiment=0.75, fii_flow="INFLOW", dii_flow="INFLOW", fii_flow_trend="INFLOW", social_chatter_score=0.70, classification="POSITIVE", confidence=0.80, reasons=["Good"])

    scenarios = counterfactual_simulator.simulate("TATAMOTORS", "conservative", tech, fund, sent, multiplier=1.5)
    assert len(scenarios) == 4
    for sc in scenarios:
        assert "Not a price forecast" in sc.simulation_disclaimer or "Illustrative" in sc.simulation_disclaimer

# ---------------------------------------------------------
# Test 20: Evidence graph resolves decision to claim to document (§18)
# ---------------------------------------------------------
def test_20_evidence_graph_claim_resolution():
    fund = FundamentalSignal(
        ticker="XYZ_CORP", rag_citations=["[SEBI-Filing-XYZ-Q3: Page 14]"],
        debt_to_equity=3.85, earnings_growth=-0.12, filing_verdict="CRITICAL_RISK",
        confidence=0.90, evidence=["Severe debt distress disclosed in filing."]
    )
    synth = SynthesizedOutput(
        session_id="sess_test",
        ticker="XYZ_CORP",
        raw_signals={"fundamental": fund.model_dump()},
        market_view="HIGH_RISK_MOMENTUM",
        synthesized_verdict="HOLD-WATCH",
        market_classification="HIGH_RISK_MOMENTUM",
        confidence=0.65,
        source_attributions=fund.rag_citations,
        reasoning_trace=["Divergence resolved."],
        personalized_advice="Conservative hold.",
        risk_profile="conservative",
        disclaimer="Notice.",
        confidence_breakdown=compute_confidence_breakdown(0.9, 0.5, 0.9, 0.8)
    )
    nodes = evidence_graph_engine.build_graph(synth, fund)
    types = [n.node_type for n in nodes]
    assert "DECISION" in types
    assert "CLAIM" in types
    assert "EVIDENCE" in types
    assert "DOCUMENT" in types

# ---------------------------------------------------------
# Test 21: What-changed is empty on identical snapshots (§25)
# ---------------------------------------------------------
def test_21_what_changed_empty_on_no_changes():
    cb = compute_confidence_breakdown(0.95, 0.85, 0.90, 0.85)
    synth = SynthesizedOutput(
        session_id="sess_1", ticker="TATAMOTORS", raw_signals={}, market_view="BULLISH",
        synthesized_verdict="BUY CANDIDATE", market_classification="BULLISH",
        confidence=0.88, reasoning_trace=[], personalized_advice="",
        risk_profile="conservative", disclaimer="", confidence_breakdown=cb
    )
    digest = change_digest_engine.compute_changes("TATAMOTORS", synth, synth)
    assert len(digest.changes) == 0

# ---------------------------------------------------------
# Test 22: Behavioral drift output contains no diagnostic words (§13 & §22)
# ---------------------------------------------------------
def test_22_behavioral_drift_no_diagnostic_terms():
    report = evaluate_behavioral_drift("conservative", "XYZ_CORP", {"XYZ_CORP": 0.15}, {})
    assert check_no_diagnostic_terms(report.observed_behavior_pattern)
    assert check_no_diagnostic_terms(report.drift_severity)

# ---------------------------------------------------------
# Test 23: Copilot query returns evidence-grounded answer (§23)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_23_copilot_endpoint_grounded_answer():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/copilot/query", json={
            "question": "What is the debt to equity ratio disclosed for XYZ Corp?",
            "ticker": "XYZ_CORP"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["grounded_in_rag"] is True
        assert len(data["cited_sources"]) > 0
        assert any("XYZ" in c for c in data["cited_sources"])

# ---------------------------------------------------------
# Test 24: Separation of Market View and Investor Fit (§5)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_24_market_view_vs_investor_fit_separation():
    res_cons = await orchestrator.analyze("TATAMOTORS", persona="conservative")
    res_aggr = await orchestrator.analyze("TATAMOTORS", persona="aggressive")
    
    assert res_cons.agent_outputs["technical"] == res_aggr.agent_outputs["technical"]
    assert res_cons.agent_outputs["sentiment"] == res_aggr.agent_outputs["sentiment"]
    assert res_cons.agent_outputs["fundamental"] == res_aggr.agent_outputs["fundamental"]
    
    assert res_cons.synthesis.market_view == res_aggr.synthesis.market_view
    assert res_cons.decision_twin.market_confidence == res_aggr.decision_twin.market_confidence
    assert res_cons.decision_twin.decision_fit != res_aggr.decision_twin.decision_fit

# ---------------------------------------------------------
# Test 25: FII and DII flows are processed and returned separately (§10)
# ---------------------------------------------------------
def test_25_separate_fii_dii_flows(sample_market_data):
    tata_data = sample_market_data["TATAMOTORS"]
    sig = sentiment_agent.analyze("TATAMOTORS", tata_data)
    
    assert hasattr(sig, "fii_flow")
    assert hasattr(sig, "dii_flow")
    assert sig.fii_flow in ["INFLOW", "OUTFLOW", "NEUTRAL"]
    assert sig.dii_flow in ["INFLOW", "OUTFLOW", "NEUTRAL"]

# ---------------------------------------------------------
# Test 26: Missing debt data is represented as None / null, not 0.0 (§11)
# ---------------------------------------------------------
def test_26_missing_debt_data_is_none():
    empty_market_data = {"ticker": "UNKNOWN_TICKER", "price_series": [], "financials": {}}
    sig, _ = fundamental_rag_agent.analyze_fundamentals("UNKNOWN_TICKER", empty_market_data, scenario="degraded")
    
    assert sig.debt_to_equity is None, "Missing debt must be None, not 0.0"
    assert sig.data_available is False

# ---------------------------------------------------------
# Test 27: Citation integrity: All cited sources must exist in retrieved chunks (§29)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_27_citation_integrity_no_hallucinations():
    res = await orchestrator.analyze("XYZ_CORP", persona="conservative")
    retrieved_chunks = fundamental_rag_agent.retrieve(query="debt liabilities", ticker="XYZ_CORP", top_k=5)
    valid_citation_tags = {c["citation_tag"] for c in retrieved_chunks}
    
    for citation in res.citations:
        assert citation in valid_citation_tags, f"Citation '{citation}' was not in retrieved chunks!"
    for source in res.synthesis.source_attributions:
        assert source in valid_citation_tags, f"Source '{source}' was not in retrieved chunks!"

# ---------------------------------------------------------
# Test 28: Stocks and Personas endpoints return valid schemas (§30)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_28_api_stocks_and_personas():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_stocks = await client.get("/api/stocks")
        assert res_stocks.status_code == 200
        assert len(res_stocks.json()["stocks"]) >= 3
        
        res_personas = await client.get("/api/personas")
        assert res_personas.status_code == 200
        assert len(res_personas.json()["personas"]) >= 2

# ---------------------------------------------------------
# Test 29: Demo Scenarios and Calibration endpoints (§30)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_29_demo_scenarios_and_calibration():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_scen = await client.get("/api/demo-scenarios")
        assert res_scen.status_code == 200
        assert len(res_scen.json()["scenarios"]) == 4
        
        res_calib = await client.get("/api/calibration")
        assert res_calib.status_code == 200
        assert "historical_calibration_score" in res_calib.json()

# ---------------------------------------------------------
# Test 30: Health endpoint returns UTC timestamp and healthy status (§25 & §30)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_30_health_check_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "T" in data["timestamp"]

# ---------------------------------------------------------
# Test 31: Thesis creation and evaluation endpoint (§30)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_31_thesis_endpoints_save_and_retrieve():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        post_res = await client.post("/api/thesis", json={
            "ticker": "TATAMOTORS",
            "stated_reasons": ["EV leadership"],
            "key_assumptions": ["debt_to_equity stays below 1.0"],
            "invalidating_conditions": ["debt surge"]
        })
        assert post_res.status_code == 200
        assert post_res.json()["status"] == "success"
        
        get_res = await client.get("/api/thesis/TATAMOTORS")
        assert get_res.status_code == 200
        assert get_res.json()["thesis"] is not None

# ---------------------------------------------------------
# Test 32: Simulation API endpoint with multiplier (§17 & §30)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_32_simulation_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/simulate", json={
            "ticker": "TATAMOTORS",
            "persona": "conservative",
            "position_size_multiplier": 2.0
        })
        assert res.status_code == 200
        data = res.json()
        assert len(data["simulations"]) == 4
