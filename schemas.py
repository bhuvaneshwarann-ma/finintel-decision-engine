from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# ---------------------------------------------------------
# Core Signal Models (§7.1)
# ---------------------------------------------------------

class TechnicalSignal(BaseModel):
    ticker: str
    rsi: float
    macd_signal: str  # e.g., "BULLISH_CROSSOVER", "BEARISH_CROSSOVER", "NEUTRAL"
    momentum_score: float
    volume_anomaly_score: float
    classification: str  # STRONG_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG_BEARISH
    confidence: float
    reasons: List[str]
    degraded: bool = False

class FundamentalSignal(BaseModel):
    ticker: str
    rag_citations: List[str]
    debt_to_equity: Optional[float] = None  # None if filing/data is unavailable (§11)
    earnings_growth: Optional[float] = None
    filing_verdict: str  # POSITIVE, NEUTRAL, CONCERNING, CRITICAL_RISK, UNKNOWN, INSUFFICIENT_EVIDENCE
    confidence: float
    evidence: List[str]
    data_available: bool = True
    degraded: bool = False
    evidence_provenance: Optional[Dict[str, Any]] = None

class SentimentSignal(BaseModel):
    ticker: str
    news_sentiment: float  # -1.0 to +1.0
    fii_flow: str = "NEUTRAL"  # INFLOW, OUTFLOW, NEUTRAL (§10)
    dii_flow: str = "NEUTRAL"  # INFLOW, OUTFLOW, NEUTRAL (§10)
    fii_flow_trend: str = "NEUTRAL"  # Preserved for backward compatibility
    social_chatter_score: float  # 0.0 to 1.0
    classification: str   # POSITIVE, NEUTRAL, NEGATIVE, HIGHLY_SPECULATIVE
    confidence: float
    reasons: List[str]
    degraded: bool = False

class BehavioralFlag(BaseModel):
    flag_type: str  # RECENCY_BIAS, FAMILIARITY_CONCENTRATION, MOMENTUM_CHASING, LOSS_AVERSION_HOLD
    trigger_reason: str
    nudge_message: str

class RiskAssessment(BaseModel):
    risk_profile: str  # conservative, aggressive
    portfolio_concentration: float
    sector_exposure: float
    risk_flags: List[str]
    risk_score: float  # 0.0 to 1.0
    personalized_constraints: List[str]
    behavioral_flags: List[BehavioralFlag] = Field(default_factory=list)

class AgentResult(BaseModel):
    agent_name: str
    status: str  # SUCCESS, DEGRADED, FAILED
    latency_ms: float
    confidence: float
    payload: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)

class DevilsAdvocateChallenge(BaseModel):
    ticker: str
    target_verdict: str
    counter_argument: str
    weakest_evidence_point: str
    severity: str  # LOW, MEDIUM, HIGH
    confidence_adjustment: float = 0.0

class ConfidenceBreakdown(BaseModel):
    data_freshness_score: float
    agent_agreement_score: float
    evidence_strength_score: float
    historical_calibration_score: float
    composite_confidence: float

class SynthesizedOutput(BaseModel):
    session_id: str
    ticker: str
    raw_signals: Dict[str, Any]
    market_view: str  # Independent raw market signal (§5: BULLISH, NEUTRAL, BEARISH, HIGH_RISK_MOMENTUM)
    synthesized_verdict: str  # BUY CANDIDATE, HOLD-WATCH, REDUCE EXPOSURE, AVOID-HIGH RISK
    market_classification: str
    confidence: float
    investor_fit_score: float = 0.5  # Independent investor suitability score (0.0 to 1.0)
    reasoning_trace: List[str]
    conflict_summary: Optional[str] = None
    source_attributions: List[str] = Field(default_factory=list)
    personalized_advice: str
    risk_profile: str
    degraded_data: bool = False
    llm_used: bool = False
    fallback_used: bool = False
    disclaimer: str
    devils_advocate_challenge: Optional[DevilsAdvocateChallenge] = None
    confidence_breakdown: ConfidenceBreakdown
    filing_freshness_warning: Optional[str] = None

# ---------------------------------------------------------
# Investor Decision Intelligence Engine Models (§18)
# ---------------------------------------------------------

class DecisionTwin(BaseModel):
    ticker: str
    persona: str
    market_confidence: float
    decision_fit: float
    evidence_quality: float
    thesis_health: float
    behavioral_risk: float
    composite_note: str

class ThesisRecord(BaseModel):
    thesis_record_id: str
    ticker: str
    user_id: str
    created_at: str
    stated_reasons: List[str]
    key_assumptions: List[str]
    invalidating_conditions: List[str]

class ThesisBreakEvent(BaseModel):
    ticker: str
    thesis_record_id: str
    triggered_at: str
    broken_assumption: str
    evidence_citation: str
    severity: str  # WATCH, WEAKENED, BROKEN
    explanation: str

class ScenarioSimulation(BaseModel):
    ticker: str
    scenario_type: str  # BEST, BASE, FAILURE, THESIS_BREAK
    assumption_changed: str
    projected_portfolio_impact_pct: float
    projected_concentration_after: float
    narrative_explanation: str
    simulation_disclaimer: str = "Simulated illustration based on demo data. Not a price forecast or guarantee."

class BehavioralDriftReport(BaseModel):
    declared_risk_profile: str
    observed_behavior_pattern: str
    drift_flags: List[BehavioralFlag] = Field(default_factory=list)
    drift_severity: str  # NONE, MILD, NOTABLE

class EvidenceNode(BaseModel):
    node_id: str
    node_type: str  # DECISION, CLAIM, EVIDENCE, DOCUMENT
    label: str
    citation_tag: Optional[str] = None
    parent_node_id: Optional[str] = None

class ChangeItem(BaseModel):
    category: str  # THESIS, RISK, MARKET
    description: str

class ChangeDigest(BaseModel):
    session_id: str
    ticker: str
    changes: List[ChangeItem] = Field(default_factory=list)

class CopilotQueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    ticker: Optional[str] = "TATAMOTORS"

class CopilotQueryResponse(BaseModel):
    answer: str
    cited_sources: List[str] = Field(default_factory=list)
    grounded_in_rag: bool = True
    ai_mode: str = "mock"

# ---------------------------------------------------------
# API Payloads
# ---------------------------------------------------------

class AnalyzeRequest(BaseModel):
    ticker: str
    persona: str = "conservative"
    scenario: str = "aligned"

class AnalyzeResponse(BaseModel):
    session_id: str
    ticker: str
    agent_outputs: Dict[str, Any]
    synthesis: SynthesizedOutput
    reasoning_trace: List[str]
    citations: List[str]
    risk_assessment: RiskAssessment
    decision_twin: DecisionTwin
    thesis_break_events: List[ThesisBreakEvent] = Field(default_factory=list)
    behavioral_drift_report: BehavioralDriftReport
    telemetry: Dict[str, Any]
    degraded_status: Dict[str, Any]

class ThesisCreateRequest(BaseModel):
    ticker: str
    user_id: str = "demo_retail_user"
    stated_reasons: List[str]
    key_assumptions: List[str]
    invalidating_conditions: List[str]

class SimulationRequest(BaseModel):
    ticker: str
    persona: str = "conservative"
    position_size_multiplier: float = 1.0
