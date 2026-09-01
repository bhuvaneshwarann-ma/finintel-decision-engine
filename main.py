import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import (
    AI_MODE, APP_ENV, VALID_TICKERS, VALID_PERSONAS,
    VALID_SCENARIOS, FILINGS_DIR, BASE_DIR
)
from schemas import (
    AnalyzeRequest, AnalyzeResponse, ThesisCreateRequest,
    ThesisRecord, SimulationRequest, ScenarioSimulation,
    EvidenceNode, ChangeDigest, DecisionTwin
)
from utils.security import validate_ticker, validate_persona, validate_scenario
from agents.orchestrator import orchestrator
from agents.counterfactual_simulator import counterfactual_simulator
from engine.evidence_graph import evidence_graph_engine
from engine.change_digest import change_digest_engine
from engine.decision_twin import decision_twin_engine
from profiling.user_profile import PERSONA_PROFILES
from logs.session_logger import session_logger
from logs.calibration_ledger import calibration_ledger

app = FastAPI(
    title="FinIntel Decision Engine API",
    description="Multi-Agent Autonomous Financial Intelligence System for Retail Investors",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = BASE_DIR / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = BASE_DIR / "static" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>FinIntel Decision Engine Dashboard</h1><p>Static UI loading...</p>")

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = BASE_DIR / "static" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>FinIntel Decision Engine Dashboard</h1><p>Static UI loading...</p>")


@app.get("/antigravity", response_class=HTMLResponse)
async def serve_antigravity():
    antigravity_file = BASE_DIR / "static" / "antigravity.html"
    if antigravity_file.exists():
        with open(antigravity_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>ANTIGRAVITY Experience</h1><p>Loading...</p>")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ai_mode": AI_MODE,
        "app_env": APP_ENV,
        "indexed_documents": len(orchestrator.market_data_cache),
        "calibration_runs": len(calibration_ledger._records)
    }

@app.get("/api/stocks")
async def get_stocks():
    stocks = []
    for ticker in VALID_TICKERS:
        m_data = orchestrator.market_data_cache.get(ticker, {})
        stocks.append({
            "ticker": ticker,
            "name": m_data.get("name", ticker),
            "sector": m_data.get("sector", "Equities"),
            "latest_close": m_data.get("price_series", [{}])[-1].get("close", 0.0) if m_data.get("price_series") else 0.0
        })
    return {"stocks": stocks}

@app.get("/api/personas")
async def get_personas():
    return {"personas": list(PERSONA_PROFILES.values())}

@app.get("/api/demo-scenarios")
async def get_demo_scenarios():
    return {
        "scenarios": [
            {
                "id": "A",
                "scenario_key": "aligned",
                "ticker": "TATAMOTORS",
                "name": "Scenario A — Aligned Bullish",
                "description": "Technical breakout + positive sentiment + balance sheet deleveraging. Expected: BUY CANDIDATE (cited).",
                "lead_demo": False
            },
            {
                "id": "B",
                "scenario_key": "conflict",
                "ticker": "XYZ_CORP",
                "name": "Scenario B — Conflicting Signals",
                "description": "Bullish technicals + Bullish sentiment + hidden debt risk in filing. Anti-majority-vote resolution.",
                "lead_demo": False
            },
            {
                "id": "C",
                "scenario_key": "degraded",
                "ticker": "TATAMOTORS",
                "name": "Scenario C — Degraded Data Feed",
                "description": "Information feeds degraded; system operates in graceful fallback with reduced confidence and clear warnings.",
                "lead_demo": False
            },
            {
                "id": "D",
                "scenario_key": "stale_behavioral",
                "ticker": "XYZ_CORP",
                "name": "Scenario D — Stale Evidence + Behavioral Nudge",
                "description": "Stale filing check + FOMO/momentum chasing bias detection + Devil's Advocate challenge. (Lead Demo)",
                "lead_demo": True
            }
        ]
    }

@app.get("/api/calibration")
async def get_calibration():
    return calibration_ledger.get_summary()

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_stock(request: AnalyzeRequest):
    ticker = validate_ticker(request.ticker)
    persona = validate_persona(request.persona)
    scenario = validate_scenario(request.scenario)
    
    response = await orchestrator.analyze(ticker=ticker, persona=persona, scenario=scenario)
    return response

@app.post("/api/thesis")
async def save_thesis_endpoint(request: ThesisCreateRequest):
    ticker = validate_ticker(request.ticker)
    import uuid
    thesis = ThesisRecord(
        thesis_record_id=f"THESIS-{ticker}-{uuid.uuid4().hex[:6]}",
        ticker=ticker,
        user_id=request.user_id,
        created_at=now_str,
        stated_reasons=request.stated_reasons,
        key_assumptions=request.key_assumptions,
        invalidating_conditions=request.invalidating_conditions
    )
    orchestrator.save_thesis(thesis)
    return {"status": "success", "thesis": thesis}

@app.get("/api/thesis/{ticker}")
async def get_thesis_endpoint(ticker: str):
    validated_ticker = validate_ticker(ticker)
    thesis = orchestrator.get_thesis(validated_ticker)
    if not thesis:
        return {"ticker": validated_ticker, "thesis": None, "break_events": []}
    
    # Check current break events
    m_data = orchestrator.market_data_cache.get(validated_ticker, {})
    from agents.technical_agent import technical_agent
    from agents.fundamental_rag_agent import fundamental_rag_agent
    from agents.sentiment_agent import sentiment_agent
    from agents.thesis_break_agent import thesis_break_agent

    tech_sig = technical_agent.analyze(validated_ticker, m_data)
    fund_sig, _ = fundamental_rag_agent.analyze_fundamentals(validated_ticker, m_data)
    sent_sig = sentiment_agent.analyze(validated_ticker, m_data)

    breaks = thesis_break_agent.evaluate_thesis(thesis, fund_sig, tech_sig, sent_sig)
    return {
        "ticker": validated_ticker,
        "thesis": thesis,
        "break_events": breaks
    }

@app.get("/api/decision-twin/{ticker}")
async def get_decision_twin(ticker: str, persona: str = "conservative"):
    val_ticker = validate_ticker(ticker)
    val_persona = validate_persona(persona)

    # Run quick analysis to fetch decision twin
    res = await orchestrator.analyze(ticker=val_ticker, persona=val_persona)
    return res.decision_twin

@app.post("/api/simulate")
async def simulate_scenarios(request: SimulationRequest):
    ticker = validate_ticker(request.ticker)
    persona = validate_persona(request.persona)
    m_data = orchestrator.market_data_cache.get(ticker, {})

    from agents.technical_agent import technical_agent
    from agents.fundamental_rag_agent import fundamental_rag_agent
    from agents.sentiment_agent import sentiment_agent
    
    tech_sig = technical_agent.analyze(ticker, m_data)
    fund_sig, _ = fundamental_rag_agent.analyze_fundamentals(ticker, m_data)
    sent_sig = sentiment_agent.analyze(ticker, m_data)

    scenarios = counterfactual_simulator.simulate(
        ticker=ticker,
        persona_id=persona,
        technical_sig=tech_sig,
        fundamental_sig=fund_sig,
        sentiment_sig=sent_sig
    )
    return {"ticker": ticker, "persona": persona, "simulations": scenarios}

@app.get("/api/evidence-graph/{session_id}")
async def get_evidence_graph(session_id: str):
    sess = session_logger.get_session(session_id)
    if not sess:
        # Generate dummy fallback or lookup latest
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    ticker = sess["ticker"]
    persona = sess["persona"]
    # Reconstruct from stored state or orchestrator
    m_data = orchestrator.market_data_cache.get(ticker, {})
    from agents.fundamental_rag_agent import fundamental_rag_agent
    fund_sig, _ = fundamental_rag_agent.analyze_fundamentals(ticker, m_data)
    
    # Build SynthesizedOutput wrapper
    from schemas import SynthesizedOutput, ConfidenceBreakdown
    synthesis = SynthesizedOutput(
        session_id=session_id,
        ticker=ticker,
        raw_signals={"technical": {}, "fundamental": fund_sig.model_dump(), "sentiment": {}},
        synthesized_verdict=sess["synthesized_verdict"],
        market_classification=sess["signal_classification"],
        confidence=sess["confidence"],
        reasoning_trace=[],
        source_attributions=fund_sig.rag_citations,
        personalized_advice="",
        risk_profile=persona,
        degraded_data=sess["degraded_data"],
        disclaimer="",
        confidence_breakdown=ConfidenceBreakdown(
            data_freshness_score=0.9, agent_agreement_score=0.8, evidence_strength_score=0.9,
            historical_calibration_score=0.85, composite_confidence=sess["confidence"]
        )
    )

    nodes = evidence_graph_engine.build_graph(synthesis, fund_sig.evidence)
    return {"session_id": session_id, "nodes": nodes}

@app.get("/api/what-changed/{ticker}")
async def get_what_changed(ticker: str, since: Optional[str] = Query(None)):
    val_ticker = validate_ticker(ticker)
    prev_session = session_logger.get_previous_session_for_ticker(val_ticker, current_session_id=since)
    current_session = session_logger.get_session(since) if since else None

    if not current_session:
        # Run fresh or mock baseline
        res = await orchestrator.analyze(ticker=val_ticker, persona="conservative")
        current_data = session_logger.get_session(res.session_id)
    else:
        current_data = current_session

    digest = change_digest_engine.compute_changes(
        ticker=val_ticker,
        current_data=current_data or {},
        previous_data=prev_session,
        since_session_id=since
    )
    return digest

class CopilotRequest(BaseModel):
    ticker: str
    query: str
    persona: Optional[str] = "conservative"

@app.post("/api/copilot")
async def copilot_query(request: CopilotRequest):
    val_ticker = validate_ticker(request.ticker)
    from agents.fundamental_rag_agent import fundamental_rag_agent
    retrieved = fundamental_rag_agent.retrieve(request.query, val_ticker, top_k=2)
    citations = [r["citation_tag"] for r in retrieved]
    evidence_texts = [r["text"] for r in retrieved]
    
    res = await orchestrator.analyze(val_ticker, persona=request.persona or "conservative")
    synth = res.synthesis
    fund = res.agent_outputs["fundamental"]
    
    q_lower = request.query.lower()
    if "debt" in q_lower or "borrowing" in q_lower or "pledge" in q_lower:
        ans = (
            f"Regulatory filing audit confirms Debt-to-Equity is {fund['debt_to_equity']:.2f}x. "
            + (f"Cited in {citations[0]}: \"{evidence_texts[0][:150]}...\"" if citations else "No fresh filing available.")
        )
    elif "margin" in q_lower or "earnings" in q_lower or "revenue" in q_lower:
        ans = (
            f"Earnings trajectory shows {fund['earnings_growth']*100:.1f}% growth with {fund['filing_verdict']} solvency grade. "
            + (f"Grounding citation: {citations[0]}." if citations else "")
        )
    elif "verdict" in q_lower or "buy" in q_lower or "recommendation" in q_lower:
        ans = (
            f"Synthesized recommendation is '{synth.synthesized_verdict}' ({synth.market_classification}) "
            f"with composite confidence of {synth.confidence*100:.1f}%. {synth.personalized_advice}"
        )
    else:
        ans = (
            f"Multi-agent synthesis for {val_ticker} indicates {synth.market_classification} market conditions. "
            f"Decision Twin shows market confidence at {res.decision_twin.market_confidence} and decision fit at {res.decision_twin.decision_fit}. "
            + (f"Key filing reference: {citations[0]}." if citations else "")
        )

    return {
        "ticker": val_ticker,
        "query": request.query,
        "answer": ans,
        "citations": citations,
        "confidence": synth.confidence
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
