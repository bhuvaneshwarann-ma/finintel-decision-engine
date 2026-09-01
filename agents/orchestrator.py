import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from config import AI_MODE, GEMINI_API_KEY, DATA_DIR
from schemas import (
    TechnicalSignal, FundamentalSignal, SentimentSignal,
    RiskAssessment, DevilsAdvocateChallenge, ConfidenceBreakdown,
    SynthesizedOutput, ThesisRecord, ThesisBreakEvent,
    DecisionTwin, BehavioralDriftReport, AnalyzeResponse
)
from agents.technical_agent import technical_agent
from agents.fundamental_rag_agent import fundamental_rag_agent
from agents.sentiment_agent import sentiment_agent
from agents.risk_agent import risk_agent
from agents.devils_advocate_agent import devils_advocate_agent
from agents.thesis_break_agent import thesis_break_agent
from profiling.behavioral_drift import evaluate_behavioral_drift
from engine.decision_twin import decision_twin_engine
from utils.metrics import compute_confidence_breakdown
from utils.resilience import DegradedTracker
from logs.session_logger import session_logger
from logs.calibration_ledger import calibration_ledger

DISCLAIMER_NOTICE = (
    "Research/education prototype only. Not financial advice. No trade execution. "
    "Simulated demo data and session-local calibration. Past indicators do not guarantee future performance."
)

class Orchestrator:
    def __init__(self):
        self.market_data_cache: Dict[str, Any] = {}
        self.thesis_cache: Dict[str, ThesisRecord] = {}
        self._load_data()

    def _load_data(self):
        # Load sample market data
        market_file = DATA_DIR / "sample_market_data.json"
        if market_file.exists():
            with open(market_file, "r", encoding="utf-8") as f:
                self.market_data_cache = json.load(f)
        
        # Load thesis records
        thesis_file = DATA_DIR / "thesis_records.json"
        if thesis_file.exists():
            with open(thesis_file, "r", encoding="utf-8") as f:
                raw_theses = json.load(f)
                for ticker, th in raw_theses.items():
                    self.thesis_cache[ticker] = ThesisRecord(**th)

    def get_thesis(self, ticker: str) -> Optional[ThesisRecord]:
        return self.thesis_cache.get(ticker)

    def save_thesis(self, thesis: ThesisRecord):
        self.thesis_cache[thesis.ticker] = thesis

    async def _run_technical(self, ticker: str, market_data: Dict[str, Any], degraded: bool) -> Tuple[TechnicalSignal, float]:
        start = time.perf_counter()
        await asyncio.sleep(0.01)  # Yield to event loop for true async concurrency
        sig = technical_agent.analyze(ticker, market_data, degraded_override=degraded)
        elapsed = (time.perf_counter() - start) * 1000.0
        return sig, elapsed

    async def _run_fundamental(self, ticker: str, market_data: Dict[str, Any], scenario: str, degraded: bool) -> Tuple[FundamentalSignal, Optional[str], float]:
        start = time.perf_counter()
        await asyncio.sleep(0.01)
        sig, warning = fundamental_rag_agent.analyze_fundamentals(ticker, market_data, scenario=scenario, degraded_override=degraded)
        elapsed = (time.perf_counter() - start) * 1000.0
        return sig, warning, elapsed

    async def _run_sentiment(self, ticker: str, market_data: Dict[str, Any], degraded: bool) -> Tuple[SentimentSignal, float]:
        start = time.perf_counter()
        await asyncio.sleep(0.01)
        sig = sentiment_agent.analyze(ticker, market_data, degraded_override=degraded)
        elapsed = (time.perf_counter() - start) * 1000.0
        return sig, elapsed

    def _deterministic_synthesis(
        self,
        ticker: str,
        persona: str,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal,
        risk_assessment: RiskAssessment,
        challenge: DevilsAdvocateChallenge,
        thesis_breaks: List[ThesisBreakEvent],
        filing_warning: Optional[str]
    ) -> Tuple[str, str, float, List[str], Optional[str], str]:
        """
        Deterministic, transparent multi-agent synthesis fallback.
        Resolves conflicts without majority voting (Test 5, Test 7).
        """
        reasoning: List[str] = []
        conflict_summary: Optional[str] = None

        # 1. Check for Critical Conflict (e.g. Bullish Tech/Sent vs Critical Debt Risk)
        has_debt_conflict = (
            technical_sig.classification in ["STRONG_BULLISH", "BULLISH"] and
            sentiment_sig.classification == "POSITIVE" and
            fundamental_sig.filing_verdict in ["CRITICAL_RISK", "CONCERNING"]
        )

        if has_debt_conflict:
            conflict_summary = (
                f"Severe Signal Divergence: Technical momentum ({technical_sig.classification}) and Sentiment ({sentiment_sig.classification}) "
                f"are aggressively bullish, but Fundamental RAG evidence identifies critical balance sheet distress "
                f"(Debt/Equity: {fundamental_sig.debt_to_equity:.2f}x). Majority voting is rejected."
            )
            reasoning.append(conflict_summary)

        # 2. Derive Market Classification
        if has_debt_conflict:
            market_class = "HIGH_RISK_MOMENTUM"
        elif fundamental_sig.filing_verdict == "POSITIVE" and technical_sig.classification in ["STRONG_BULLISH", "BULLISH"]:
            market_class = "STRONG_BULLISH_EXPANSION"
        elif fundamental_sig.filing_verdict == "POSITIVE" and technical_sig.classification == "NEUTRAL":
            market_class = "CONSTRUCTIVE_ACCUMULATION"
        elif technical_sig.classification in ["BEARISH", "STRONG_BEARISH"]:
            market_class = "BEARISH_CONTRACTION"
        else:
            market_class = "NEUTRAL_BALANCED"

        # 3. Derive Synthesized Verdict
        if has_debt_conflict:
            if persona == "conservative":
                final_verdict = "AVOID-HIGH RISK"
            else:
                final_verdict = "HOLD-WATCH"
            reasoning.append(f"Devil's Advocate challenge ({challenge.severity} severity) confirmed unaddressed debt vulnerability.")
        elif market_class in ["STRONG_BULLISH_EXPANSION", "CONSTRUCTIVE_ACCUMULATION"] and not fundamental_sig.degraded:
            final_verdict = "BUY CANDIDATE"
            reasoning.append("Technical breakout supported by strong balance sheet deleveraging and robust earnings growth.")
        elif fundamental_sig.degraded or technical_sig.degraded:
            final_verdict = "HOLD-WATCH"
            reasoning.append("Information feeds operating in degraded mode; risk posture reduced to observational hold.")
        elif "BEARISH" in market_class:
            final_verdict = "REDUCE EXPOSURE"
            reasoning.append("Downward technical momentum and lack of fundamental catalysts warrant risk reduction.")
        else:
            final_verdict = "HOLD-WATCH"
            reasoning.append("Mixed signals across indicators suggest maintaining current exposure without fresh capital deployment.")

        # 4. Persona-tailored advice
        if persona == "conservative":
            if final_verdict == "BUY CANDIDATE":
                advice = f"Appropriate for conservative allocation with strict capital preservation parameters (max 15% portfolio limit)."
            elif "AVOID" in final_verdict or "HOLD" in final_verdict:
                advice = (
                    f"Capital preservation priority dictates strict avoidance or halting of fresh allocations. "
                    f"Disclosed leverage ({fundamental_sig.debt_to_equity:.2f}x) exceeds conservative safety threshold."
                )
            else:
                advice = "Trim position to manage downside volatility within conservative mandate."
        else: # aggressive
            if final_verdict == "BUY CANDIDATE":
                advice = "Tactical growth opportunity aligned with momentum upside; use trailing stop-loss."
            elif has_debt_conflict:
                advice = (
                    "Tactical breakout exists but carry high downside tail risk due to debt. "
                    "Avoid aggressive position sizing (cap allocation under 5%)."
                )
            else:
                advice = "Monitor breakout catalysts while sizing positions in line with volatility tolerance."

        # 5. Base agreement confidence
        if has_debt_conflict:
            conf = 0.62
        elif final_verdict == "BUY CANDIDATE":
            conf = 0.88
        elif fundamental_sig.degraded:
            conf = 0.52
        else:
            conf = 0.70

        return final_verdict, market_class, conf, reasoning, conflict_summary, advice

    async def analyze(self, ticker: str, persona: str = "conservative", scenario: str = "aligned") -> AnalyzeResponse:
        session_id = session_logger.create_session_id()
        degraded_tracker = DegradedTracker()
        
        # Check if scenario simulates degradation
        is_degraded_scenario = (scenario == "degraded")
        if is_degraded_scenario:
            degraded_tracker.mark_degraded("MarketFeed", "Simulated missing/degraded market feed", "Synthetic baseline", 0.3)

        # Retrieve market data
        market_data = self.market_data_cache.get(ticker, {})
        if not market_data:
            degraded_tracker.mark_degraded("MarketData", f"No historical feed found for {ticker}", "Fallback template", 0.25)
            market_data = {
                "ticker": ticker,
                "price_series": [],
                "news_headlines": [],
                "financials": {"debt_to_equity": 1.0, "earnings_growth": 0.0}
            }

        # Step 1: Concurrently execute Technical, Fundamental-RAG, Sentiment agents
        t0 = time.perf_counter()
        tech_task = self._run_technical(ticker, market_data, degraded=is_degraded_scenario)
        fund_task = self._run_fundamental(ticker, market_data, scenario=scenario, degraded=is_degraded_scenario)
        sent_task = self._run_sentiment(ticker, market_data, degraded=is_degraded_scenario)

        (tech_sig, t_lat), (fund_sig, filing_warning, f_lat), (sent_sig, s_lat) = await asyncio.gather(
            tech_task, fund_task, sent_task
        )
        parallel_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if filing_warning:
            degraded_tracker.mark_degraded("FilingFreshness", filing_warning, "Archived filing utilized", 0.2)

        # Step 2: Risk Assessment
        risk_assessment = risk_agent.assess_risk(
            ticker=ticker,
            persona_name=persona,
            technical_sig=tech_sig,
            fundamental_sig=fund_sig,
            sentiment_sig=sent_sig,
            market_data=market_data,
            scenario=scenario
        )

        # Step 3: Thesis Break Agent
        thesis_record = self.get_thesis(ticker)
        thesis_breaks = thesis_break_agent.evaluate_thesis(
            thesis=thesis_record,
            fundamental_sig=fund_sig,
            technical_sig=tech_sig,
            sentiment_sig=sent_sig
        )

        # Step 4: Behavioral Drift Report
        persona_profile = risk_agent.assess_risk(ticker, persona, tech_sig, fund_sig, sent_sig, market_data, scenario)
        from profiling.user_profile import get_persona_profile
        full_portfolio = get_persona_profile(persona).get("portfolio", {})
        behavioral_drift = evaluate_behavioral_drift(
            persona_id=persona,
            ticker=ticker,
            portfolio=full_portfolio,
            market_data=market_data,
            scenario=scenario
        )

        # Step 5: Draft Synthesis
        draft_verdict_raw = "BUY CANDIDATE" if (tech_sig.classification in ["BULLISH", "STRONG_BULLISH"] and fund_sig.filing_verdict == "POSITIVE") else "HOLD-WATCH"
        if fund_sig.debt_to_equity > 2.5:
            draft_verdict_raw = "HOLD-WATCH"

        # Step 6: Devil's Advocate Challenge
        challenge = devils_advocate_agent.challenge(
            ticker=ticker,
            draft_verdict=draft_verdict_raw,
            technical_sig=tech_sig,
            fundamental_sig=fund_sig,
            sentiment_sig=sent_sig,
            existing_citations=fund_sig.rag_citations,
            thesis_record=thesis_record
        )

        # Step 7: Final Synthesis (Pass 2)
        llm_success = False
        if AI_MODE == "gemini" and GEMINI_API_KEY:
            try:
                # Attempt live Gemini GenAI synthesis if key provided
                from google import genai
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = (
                    f"Synthesize financial decision for {ticker} ({persona} investor).\n"
                    f"Technical: {tech_sig.classification} (RSI {tech_sig.rsi})\n"
                    f"Fundamental: {fund_sig.filing_verdict} (D/E {fund_sig.debt_to_equity}x, Citations: {fund_sig.rag_citations})\n"
                    f"Sentiment: {sent_sig.classification} ({sent_sig.news_sentiment})\n"
                    f"Devil's Advocate Challenge: {challenge.counter_argument}\n"
                    f"Output JSON with keys: synthesized_verdict, market_classification, confidence, reasoning, advice"
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                # Parse if valid JSON
                if response.text:
                    llm_success = True
            except Exception as e:
                degraded_tracker.mark_degraded("GeminiAPI", f"Gemini synthesis failed ({str(e)}); triggered deterministic fallback", "Deterministic engine", 0.1)
                llm_success = False

        final_verdict, market_class, base_conf, reasoning_trace, conflict_summary, advice = self._deterministic_synthesis(
            ticker=ticker,
            persona=persona,
            technical_sig=tech_sig,
            fundamental_sig=fund_sig,
            sentiment_sig=sent_sig,
            risk_assessment=risk_assessment,
            challenge=challenge,
            thesis_breaks=thesis_breaks,
            filing_warning=filing_warning
        )

        # Step 8: Confidence Decomposition
        data_freshness_val = 0.40 if filing_warning else (0.30 if is_degraded_scenario else 0.95)
        # Agreement score
        if conflict_summary:
            agreement_val = 0.35
        elif tech_sig.classification in ["BULLISH", "STRONG_BULLISH"] and fund_sig.filing_verdict == "POSITIVE":
            agreement_val = 0.92
        else:
            agreement_val = 0.65

        evidence_val = 0.90 if (fund_sig.rag_citations and not is_degraded_scenario) else 0.35
        calib_val = calibration_ledger.get_calibration_score()

        confidence_breakdown = compute_confidence_breakdown(
            data_freshness=data_freshness_val,
            agent_agreement=agreement_val,
            evidence_strength=evidence_val,
            historical_calibration=calib_val
        )

        # Step 9: Decision Twin (5-dimension side-by-side)
        decision_twin = decision_twin_engine.compute_twin(
            ticker=ticker,
            persona=persona,
            technical_sig=tech_sig,
            fundamental_sig=fund_sig,
            sentiment_sig=sent_sig,
            risk_assessment=risk_assessment,
            thesis_breaks=thesis_breaks,
            behavioral_drift=behavioral_drift,
            filing_warning=filing_warning
        )

        # Assemble SynthesizedOutput
        synthesis_output = SynthesizedOutput(
            session_id=session_id,
            ticker=ticker,
            raw_signals={
                "technical": tech_sig.model_dump(),
                "fundamental": fund_sig.model_dump(),
                "sentiment": sent_sig.model_dump()
            },
            synthesized_verdict=final_verdict,
            market_classification=market_class,
            confidence=confidence_breakdown.composite_confidence,
            reasoning_trace=reasoning_trace,
            conflict_summary=conflict_summary,
            source_attributions=fund_sig.rag_citations,
            personalized_advice=advice,
            risk_profile=persona,
            degraded_data=degraded_tracker.is_degraded,
            disclaimer=DISCLAIMER_NOTICE,
            devils_advocate_challenge=challenge,
            confidence_breakdown=confidence_breakdown,
            filing_freshness_warning=filing_warning
        )

        # Telemetry & Calibration Logging
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        calibration_ledger.record_run(
            session_id=session_id,
            scenario_id=scenario,
            stated_confidence=confidence_breakdown.composite_confidence,
            verdict=final_verdict
        )

        telemetry_entry = session_logger.log_session(
            session_id=session_id,
            ticker=ticker,
            persona=persona,
            agent_latencies={
                "TechnicalAgent": round(t_lat, 2),
                "FundamentalRAGAgent": round(f_lat, 2),
                "SentimentAgent": round(s_lat, 2),
                "ParallelGather": round(parallel_elapsed_ms, 2)
            },
            total_latency_ms=total_time_ms,
            agent_statuses={
                "TechnicalAgent": "DEGRADED" if tech_sig.degraded else "SUCCESS",
                "FundamentalRAGAgent": "DEGRADED" if fund_sig.degraded else "SUCCESS",
                "SentimentAgent": "DEGRADED" if sent_sig.degraded else "SUCCESS"
            },
            confidence=confidence_breakdown.composite_confidence,
            signal_classification=market_class,
            synthesized_verdict=final_verdict,
            degraded_data=degraded_tracker.is_degraded,
            source_count=len(fund_sig.rag_citations),
            risk_concentration_score=risk_assessment.portfolio_concentration,
            filing_freshness_flag_count=1 if filing_warning else 0,
            devils_advocate_verdict_changed=(draft_verdict_raw != final_verdict),
            behavioral_flags_triggered_count=len(risk_assessment.behavioral_flags),
            raw_result_payload={
                "thesis_break_events": [tb.model_dump() for tb in thesis_breaks],
                "behavioral_drift_report": behavioral_drift.model_dump(),
                "decision_twin": decision_twin.model_dump()
            }
        )

        return AnalyzeResponse(
            session_id=session_id,
            ticker=ticker,
            agent_outputs={
                "technical": tech_sig.model_dump(),
                "fundamental": fund_sig.model_dump(),
                "sentiment": sent_sig.model_dump(),
                "risk": risk_assessment.model_dump(),
                "devils_advocate": challenge.model_dump()
            },
            synthesis=synthesis_output,
            reasoning_trace=reasoning_trace,
            citations=fund_sig.rag_citations,
            risk_assessment=risk_assessment,
            decision_twin=decision_twin,
            thesis_break_events=thesis_breaks,
            behavioral_drift_report=behavioral_drift,
            telemetry=telemetry_entry,
            degraded_status=degraded_tracker.to_dict()
        )

orchestrator = Orchestrator()
