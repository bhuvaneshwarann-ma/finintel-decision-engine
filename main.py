import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, EmailStr
from fastapi import FastAPI, HTTPException, Query, Request, Depends, status
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
    EvidenceNode, ChangeDigest, DecisionTwin,
    CopilotQueryRequest, CopilotQueryResponse
)
from auth.models import (
    UserRegisterRequest, UserLoginRequest, TokenResponse,
    UserResponse, UserProfile, UserProfileUpdateRequest
)
from auth.database import auth_db
from auth.auth_service import auth_service
from auth.dependencies import get_current_user, get_optional_current_user
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
    description="Multi-Agent Autonomous Financial Intelligence System for Retail Investors (HACKVERSE 2026 PS-01)",
    version="3.1.0"
)

# Security Headers Middleware (§22)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response

# Standard developer and local demo CORS configuration (§23)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = BASE_DIR / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# =========================================================
# PUBLIC UI & STATIC ENDPOINTS
# =========================================================
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

@app.get("/api/tickers/live")
async def get_live_tickers():
    """
    Returns real-time ticking prices with micro-fluctuations and market state for all active tickers.
    """
    import random, time
    now_ts = int(time.time())
    live_tickers = []
    
    for ticker in VALID_TICKERS:
        m_data = orchestrator.market_data_cache.get(ticker, {})
        base_close = m_data.get("price_series", [{}])[-1].get("close", 1000.0) if m_data.get("price_series") else 1000.0
        
        # Real-time micro-fluctuation simulation seeded dynamically by time window & ticker
        seed = (now_ts // 2) + abs(hash(ticker)) % 10000
        random.seed(seed)
        delta_pct = (random.random() - 0.47) * 1.8  # Real-time dynamic variance
        current_price = round(base_close * (1 + delta_pct / 100.0), 2)
        change_amt = round(current_price - base_close, 2)
        
        live_tickers.append({
            "ticker": ticker,
            "name": m_data.get("name", ticker),
            "sector": m_data.get("sector", "Equities"),
            "base_close": base_close,
            "current_price": current_price,
            "change_pct": round(delta_pct, 2),
            "change_amount": change_amt,
            "is_positive": change_amt >= 0,
            "volume": m_data.get("price_series", [{}])[-1].get("volume", 5000000)
        })
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "tickers": live_tickers}

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

# =========================================================
# AUTHENTICATION ENDPOINTS (§7)
# =========================================================
@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest):
    """
    Registers a new investor account.
    Hashes password with Argon2 and creates an isolated user profile (§4 & §7).
    """
    existing_user = auth_db.get_user_by_email(str(request.email))
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered. Please login or use a different email."
        )

    password_hash = auth_service.hash_password(request.password)
    user_id = f"usr_{uuid.uuid4().hex[:12]}"

    user = auth_db.create_user(
        user_id=user_id,
        email=str(request.email),
        password_hash=password_hash
    )

    return {
        "message": "Registration successful",
        "user": {
            "id": user["id"],
            "email": user["email"]
        }
    }

@app.post("/auth/login", response_model=TokenResponse)
async def login_user(request: UserLoginRequest):
    """
    Authenticates user credentials and issues a signed JWT token (§7).
    Guarded with in-memory brute-force rate limiter (§24).
    """
    # 1. Check rate limit
    if not auth_service.check_rate_limit(str(request.email)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please wait 5 minutes before trying again."
        )

    # 2. Lookup user
    user = auth_db.get_user_by_email(str(request.email))
    if not user:
        auth_service.record_failed_login(str(request.email))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # 3. Verify Argon2 password hash
    is_valid = auth_service.verify_password(request.password, user["password_hash"])
    if not is_valid:
        auth_service.record_failed_login(str(request.email))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # 4. Check active status
    if not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated."
        )

    # 5. Success -> reset rate limit & issue JWT
    auth_service.reset_failed_login(str(request.email))
    access_token, expires_in = auth_service.create_access_token(user["id"], user["email"])

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in
    )

@app.post("/auth/logout")
async def logout_user():
    """
    Disposes client authentication session (§19).
    """
    return {"message": "Logout successful. Client token disposed."}

@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Returns verified profile identity for the authenticated user (§7 & §8).
    Never exposes password_hash.
    """
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        created_at=current_user["created_at"],
        is_active=current_user["is_active"],
        display_name=current_user.get("display_name")
    )

# =========================================================
# USER PROFILE ENDPOINTS (§13 & §14)
# =========================================================
@app.get("/api/profile", response_model=UserProfile)
async def get_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Retrieves isolated risk profile and portfolio settings for the authenticated user (§14).
    """
    prof = auth_db.get_user_profile(current_user["id"])
    return UserProfile(
        user_id=current_user["id"],
        email=current_user["email"],
        risk_profile=prof["risk_profile"],
        portfolio_concentration=prof["portfolio_concentration"],
        preferences=prof.get("preferences", {})
    )

@app.put("/api/profile", response_model=UserProfile)
async def update_profile(
    request: UserProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Updates the authenticated user's isolated risk profile (§14).
    """
    updated = auth_db.update_user_profile(
        user_id=current_user["id"],
        risk_profile=request.risk_profile,
        portfolio_concentration=request.portfolio_concentration,
        preferences=request.preferences
    )
    return UserProfile(
        user_id=current_user["id"],
        email=current_user["email"],
        risk_profile=updated["risk_profile"],
        portfolio_concentration=updated["portfolio_concentration"],
        preferences=updated.get("preferences", {})
    )

# =========================================================
# PROTECTED FINANCIAL INTELLIGENCE ENDPOINTS (§9, §10, §11, §12)
# =========================================================
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_stock(
    request: AnalyzeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Protected stock analysis endpoint (§9 & §10).
    Uses authenticated user ID for isolated telemetry and thesis checks.
    """
    ticker = validate_ticker(request.ticker)
    persona = validate_persona(request.persona)
    scenario = validate_scenario(request.scenario)
    
    response = await orchestrator.analyze(
        ticker=ticker,
        persona=persona,
        scenario=scenario,
        user_id=current_user["id"]
    )
    return response

@app.post("/api/thesis")
async def save_thesis_endpoint(
    request: ThesisCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Saves an investment thesis strictly scoped to the authenticated user (§11).
    Ignores any client-supplied user_id.
    """
    ticker = validate_ticker(request.ticker)
    now_str = datetime.now(timezone.utc).isoformat()
    thesis_id = f"THESIS-{ticker}-{uuid.uuid4().hex[:6]}"
    
    thesis = ThesisRecord(
        thesis_record_id=thesis_id,
        ticker=ticker,
        user_id=current_user["id"],
        created_at=now_str,
        stated_reasons=request.stated_reasons,
        key_assumptions=request.key_assumptions,
        invalidating_conditions=request.invalidating_conditions
    )
    orchestrator.save_thesis(thesis, user_id=current_user["id"])
    return {"status": "success", "thesis": thesis}

@app.get("/api/thesis/{ticker}")
async def get_thesis_endpoint(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Retrieves thesis strictly belonging to the authenticated user (§11).
    Never exposes other users' theses.
    """
    validated_ticker = validate_ticker(ticker)
    thesis = orchestrator.get_thesis(validated_ticker, user_id=current_user["id"])
    
    if not thesis or thesis.user_id != current_user["id"]:
        return {"ticker": validated_ticker, "thesis": None, "break_events": []}
    
    # Check break events for current user's thesis
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
async def get_decision_twin(
    ticker: str,
    persona: str = "conservative",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    val_ticker = validate_ticker(ticker)
    val_persona = validate_persona(persona)

    res = await orchestrator.analyze(ticker=val_ticker, persona=val_persona, user_id=current_user["id"])
    return res.decision_twin

@app.post("/api/simulate")
async def simulate_scenarios(
    request: SimulationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
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
        sentiment_sig=sent_sig,
        multiplier=request.position_size_multiplier
    )
    return {"ticker": ticker, "persona": persona, "simulations": scenarios}

@app.get("/api/evidence-graph/{session_id}")
async def get_evidence_graph(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Evidence graph endpoint scoped to authenticated user sessions (§12).
    Returns HTTP 404 if session belongs to another user.
    """
    sess = session_logger.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    # Verify user ownership if session recorded a user_id
    if sess.get("user_id") and sess.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    ticker = sess["ticker"]
    persona = sess["persona"]
    m_data = orchestrator.market_data_cache.get(ticker, {})
    from agents.fundamental_rag_agent import fundamental_rag_agent
    fund_sig, _ = fundamental_rag_agent.analyze_fundamentals(ticker, m_data)
    
    from schemas import SynthesizedOutput, ConfidenceBreakdown
    synthesis = SynthesizedOutput(
        session_id=session_id,
        ticker=ticker,
        raw_signals={"technical": {}, "fundamental": fund_sig.model_dump(), "sentiment": {}},
        market_view=sess.get("signal_classification", "NEUTRAL"),
        synthesized_verdict=sess["synthesized_verdict"],
        market_classification=sess["signal_classification"],
        confidence=sess["confidence"],
        reasoning_trace=["Retrieved from verified session audit trail."],
        personalized_advice="Archived session provenance node tree.",
        risk_profile=persona,
        disclaimer="Audited session trace.",
        confidence_breakdown=ConfidenceBreakdown(
            data_freshness_score=0.95,
            agent_agreement_score=0.85,
            evidence_strength_score=0.90,
            historical_calibration_score=0.85,
            composite_confidence=sess["confidence"]
        )
    )
    graph = evidence_graph_engine.build_graph(synthesis, fund_sig)
    return {"session_id": session_id, "nodes": graph}

@app.get("/api/what-changed/{ticker}")
async def get_what_changed(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    val_ticker = validate_ticker(ticker)
    from schemas import SynthesizedOutput, ConfidenceBreakdown, TechnicalSignal, FundamentalSignal, SentimentSignal
    m_data = orchestrator.market_data_cache.get(val_ticker, {})
    from agents.technical_agent import technical_agent
    from agents.fundamental_rag_agent import fundamental_rag_agent
    from agents.sentiment_agent import sentiment_agent

    tech_sig = technical_agent.analyze(val_ticker, m_data)
    fund_sig, _ = fundamental_rag_agent.analyze_fundamentals(val_ticker, m_data)
    sent_sig = sentiment_agent.analyze(val_ticker, m_data)

    current_synth = SynthesizedOutput(
        session_id="change_digest_current",
        ticker=val_ticker,
        raw_signals={"technical": tech_sig.model_dump(), "fundamental": fund_sig.model_dump(), "sentiment": sent_sig.model_dump()},
        market_view=tech_sig.classification,
        synthesized_verdict="BUY CANDIDATE" if fund_sig.filing_verdict == "POSITIVE" else "HOLD-WATCH",
        market_classification="BALANCED",
        confidence=0.80,
        reasoning_trace=["Baseline snapshot."],
        personalized_advice="Observation state.",
        risk_profile="conservative",
        disclaimer="Demo digest.",
        confidence_breakdown=ConfidenceBreakdown(
            data_freshness_score=0.95,
            agent_agreement_score=0.85,
            evidence_strength_score=0.90,
            historical_calibration_score=0.85,
            composite_confidence=0.80
        )
    )

    digest = change_digest_engine.compute_changes(val_ticker, current_synth, current_synth)
    return digest

@app.post("/api/copilot/query", response_model=CopilotQueryResponse)
async def query_copilot(
    request: CopilotQueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Evidence-Grounded Research Copilot & Investment Advisory Assistant (§23).
    Answers any stock, financial metric, and retail investment strategy inquiry.
    """
    from agents.copilot_engine import copilot_engine
    return copilot_engine.query(question=request.question, ticker_context=request.ticker)

