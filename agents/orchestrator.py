import asyncio
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from pydantic import ValidationError

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
        market_file = DATA_DIR / "sample_market_data.json"
        if market_file.exists():
            with open(market_file, "r", encoding="utf-8") as f:
                self.market_data_cache = json.load(f)
        
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
        await asyncio.sleep(0.015)  # True async concurrency yield point
        sig = technical_agent.analyze(ticker, market_data, degraded_override=degraded)
        elapsed = (time.perf_counter() - start) * 1000.0
        return sig, elapsed

    async def _run_fundamental(self, ticker: str, market_data: Dict[str, Any], scenario: str, degraded: bool) -> Tuple[FundamentalSignal, Optional[str], float]:
        start = time.perf_counter()
        await asyncio.sleep(0.015)
        sig, warning = fundamental_rag_agent.analyze_fundamentals(ticker, market_data, scenario=scenario, degraded_override=degraded)
        elapsed = (time.perf_counter() - start) * 1000.0
        return sig, warning, elapsed

    async def _run_sentiment(self, ticker: str, market_data: Dict[str, Any], degraded: bool) -> Tuple[SentimentSignal, float]:
        start = time.perf_counter()
        await asyncio.sleep(0.015)
        sig = sentiment_agent.analyze(ticker, market_data, degraded_override=degraded)
        elapsed = (time.perf_counter() - start) * 1000.0
        return sig, elapsed

    def _compute_market_view(
        self,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal
    ) -> Tuple[str, str, float]:
        """
        Computes the independent MARKET VIEW (§5).
        Remains completely identical regardless of which user persona evaluates it.
        """
        has_debt_conflict = (
            technical_sig.classification in ["STRONG_BULLISH", "BULLISH"] and
            sentiment_sig.classification == "POSITIVE" and
            fundamental_sig.filing_verdict in ["CRITICAL_RISK", "CONCERNING"]
        )

        if has_debt_conflict:
            market_view = "HIGH_RISK_MOMENTUM"
            market_class = "HIGH_RISK_MOMENTUM"
            market_conf = 0.65
        elif fundamental_sig.filing_verdict == "POSITIVE" and technical_sig.classification in ["STRONG_BULLISH", "BULLISH"]:
            market_view = "BULLISH"
            market_class = "STRONG_BULLISH_EXPANSION"
            market_conf = 0.88
        elif fundamental_sig.filing_verdict == "POSITIVE" and technical_sig.classification == "NEUTRAL":
            market_view = "CONSTRUCTIVE_ACCUMULATION"
            market_class = "CONSTRUCTIVE_ACCUMULATION"
            market_conf = 0.72
        elif technical_sig.classification in ["BEARISH", "STRONG_BEARISH"]:
            market_view = "BEARISH"
            market_class = "BEARISH_CONTRACTION"
            market_conf = 0.75
        elif fundamental_sig.degraded or not fundamental_sig.data_available:
            market_view = "NEUTRAL"
            market_class = "NEUTRAL_BALANCED"
            market_conf = 0.40
        else:
            market_view = "NEUTRAL"
            market_class = "NEUTRAL_BALANCED"
            market_conf = 0.60

        return market_view, market_class, market_conf

    def _compute_investor_decision_score(
        self,
        market_score: float,
        persona: str,
        risk_assessment: RiskAssessment,
        fundamental_sig: FundamentalSignal,
        challenge: DevilsAdvocateChallenge
    ) -> Tuple[float, str]:
        """
        Mathematical persona decision scoring model (§6):
        decision_score = market_score * w_m - concentration_pen * w_c - debt_pen * w_d - vol_pen * w_v + challenge_adj
        """
        w_market = 0.50
        w_conc = 0.20
        w_debt = 0.20
        w_vol = 0.10

        conc_penalty = 1.0 if risk_assessment.portfolio_concentration >= 0.15 else (risk_assessment.portfolio_concentration / 0.15)
        
        debt_penalty = 0.0
        if fundamental_sig.debt_to_equity is not None:
            if persona == "conservative":
                debt_penalty = min(1.0, max(0.0, (fundamental_sig.debt_to_equity - 1.0) / 2.0))
            else:
                debt_penalty = min(1.0, max(0.0, (fundamental_sig.debt_to_equity - 2.5) / 2.5))
        elif not fundamental_sig.data_available:
            debt_penalty = 0.5 if persona == "conservative" else 0.2

        vol_penalty = 0.5 if len(risk_assessment.risk_flags) > 1 else 0.1

        fit_score = max(0.0, min(1.0, 
            (market_score * w_market) 
            - (conc_penalty * w_conc) 
            - (debt_penalty * w_debt) 
            - (vol_penalty * w_vol) 
            + (challenge.confidence_adjustment)
        ))

        # Map fit_score to personalized action
        if persona == "conservative":
            if debt_penalty > 0.6:
                action = "AVOID-HIGH RISK"
            elif conc_penalty >= 1.0:
                action = "HOLD-WATCH"  # Target ceiling reached, maintain hold
            elif fit_score >= 0.50:
                action = "BUY CANDIDATE"
            elif fit_score >= 0.30:
                action = "HOLD-WATCH"
            else:
                action = "AVOID-HIGH RISK"
        else: # aggressive
            if fit_score >= 0.50:
                action = "BUY CANDIDATE" if debt_penalty < 0.7 else "HOLD-WATCH"
            elif fit_score >= 0.28:
                action = "HOLD-WATCH"
            else:
                action = "REDUCE EXPOSURE"

        return round(fit_score, 2), action

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
    ) -> Tuple[str, str, str, float, float, List[str], Optional[str], str]:
        """
        Deterministic, transparent multi-agent synthesis fallback.
        Resolves conflicts without majority voting (§7) and separates Market View from Investor Fit (§5).
        """
        reasoning: List[str] = []
        conflict_summary: Optional[str] = None

        # 1. Market View calculation
        market_view, market_class, market_conf = self._compute_market_view(technical_sig, fundamental_sig, sentiment_sig)

        # 2. Conflict Detection (§7)
        has_debt_conflict = (
            technical_sig.classification in ["STRONG_BULLISH", "BULLISH"] and
            sentiment_sig.classification == "POSITIVE" and
            fundamental_sig.filing_verdict in ["CRITICAL_RISK", "CONCERNING"]
        )

        if has_debt_conflict:
            de_str = f"{fundamental_sig.debt_to_equity:.2f}x" if fundamental_sig.debt_to_equity is not None else "High"
            conflict_summary = (
                f"CONFLICT DETECTED: Technical momentum ({technical_sig.classification}) and Sentiment ({sentiment_sig.classification}) "
                f"are bullish, but Fundamental RAG evidence identifies critical balance sheet distress (Debt/Equity: {de_str}). "
                f"Majority voting is rejected in favor of debt solvency weighting."
            )
            reasoning.append(conflict_summary)

        # 3. Investor Decision Score & Personalized Verdict (§5 & §6)
        investor_fit_score, final_verdict = self._compute_investor_decision_score(
            market_score=market_conf,
            persona=persona,
            risk_assessment=risk_assessment,
            fundamental_sig=fundamental_sig,
            challenge=challenge
        )

        if challenge.severity in ["HIGH", "MEDIUM"]:
            reasoning.append(f"Devil's Advocate challenge ({challenge.severity} severity, {challenge.confidence_adjustment:+.2f} adj) stress-tested the draft assumptions.")

        # 4. Reasonings & Advice
        if fundamental_sig.degraded or not fundamental_sig.data_available:
            final_verdict = "HOLD-WATCH"
            reasoning.append("Information feeds operating in degraded mode; insufficient evidence for fresh allocation.")
            advice = "Insufficient regulatory evidence available. Maintain observational hold until verified filings are published."
        elif persona == "conservative":
            if "AVOID" in final_verdict:
                de_val = f"{fundamental_sig.debt_to_equity:.2f}x" if fundamental_sig.debt_to_equity is not None else "unverified"
                advice = (
                    f"Capital preservation priority mandates strict avoidance of fresh allocations. "
                    f"Disclosed leverage ({de_val}) breaches conservative safety limits."
                )
            elif "HOLD" in final_verdict:
                advice = "Position allocation approaches persona ceiling; maintain defensive hold within conservative mandate."
            elif final_verdict == "BUY CANDIDATE":
                advice = "Favorable market breakout supported by verified balance sheet deleveraging. Cap position at 15% maximum weight."
            else:
                advice = "Maintain defensive posture within conservative mandate."
        else: # aggressive
            if has_debt_conflict:
                advice = (
                    "Tactical breakout exists but carries severe downside tail risk due to debt leverage. "
                    "Avoid aggressive sizing (cap allocation under 5% with strict stop-loss)."
                )
            elif final_verdict == "BUY CANDIDATE":
                advice = "Tactical growth opportunity aligned with momentum upside; use trailing stop-loss."
            else:
                advice = "Monitor breakout catalysts while sizing positions in line with volatility tolerance."

        return market_view, market_class, final_verdict, market_conf, investor_fit_score, reasoning, conflict_summary, advice

    async def _live_gemini_synthesis(
        self,
        ticker: str,
        persona: str,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal,
        risk_assessment: RiskAssessment,
        challenge: DevilsAdvocateChallenge
    ) -> Optional[Dict[str, Any]]:
        """
        Executes real Gemini GenAI structured synthesis if API key is provided (§4).
        """
        if not GEMINI_API_KEY:
            return None

        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = (
                f"You are the senior synthesis engine for FinIntelligence AI.\n"
                f"Stock: {ticker} | Investor: {persona}\n"
                f"Technical: {technical_sig.classification} (RSI {technical_sig.rsi}, MACD {technical_sig.macd_signal})\n"
                f"Fundamental RAG: {fundamental_sig.filing_verdict} (D/E: {fundamental_sig.debt_to_equity}x, Citations: {fundamental_sig.rag_citations})\n"
                f"Sentiment: {sentiment_sig.classification} (FII: {sentiment_sig.fii_flow}, DII: {sentiment_sig.dii_flow})\n"
                f"Devil's Advocate Challenge: {challenge.counter_argument}\n"
                f"Return JSON with exact keys: market_view, synthesized_verdict, market_classification, personalized_advice, reasoning"
            )
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt
            )
            if response and response.text:
                clean_text = response.text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:-3].strip()
                elif clean_text.startswith("```"):
                    clean_text = clean_text[3:-3].strip()
                return json.loads(clean_text)
        except Exception:
            return None
        return None

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
                "financials": {"debt_to_equity": None, "earnings_growth": None}
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
        if fund_sig.debt_to_equity is not None and fund_sig.debt_to_equity > 2.5:
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

        # Step 7: Synthesis (Live Gemini or Deterministic Fallback)
        llm_used = False
        fallback_used = False
        gemini_result = None

        if AI_MODE == "gemini" and GEMINI_API_KEY:
            gemini_result = await self._live_gemini_synthesis(
                ticker=ticker,
                persona=persona,
                technical_sig=tech_sig,
                fundamental_sig=fund_sig,
                sentiment_sig=sent_sig,
                risk_assessment=risk_assessment,
                challenge=challenge
            )
            if gemini_result:
                llm_used = True
            else:
                fallback_used = True
                degraded_tracker.mark_degraded("GeminiAPI", "Gemini synthesis timed out or failed; deterministic engine activated", "Deterministic Fallback", 0.1)
        else:
            fallback_used = True

        (
            market_view,
            market_class,
            final_verdict,
            market_conf,
            investor_fit_score,
            reasoning_trace,
            conflict_summary,
            advice
        ) = self._deterministic_synthesis(
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

        if gemini_result and "personalized_advice" in gemini_result:
            advice = gemini_result["personalized_advice"]
            if isinstance(gemini_result.get("reasoning"), list):
                reasoning_trace.extend(gemini_result["reasoning"])

        # Step 8: Confidence Decomposition (Sum of weights == 1.00 strictly)
        data_freshness_val = 0.40 if filing_warning else (0.20 if is_degraded_scenario else 0.95)
        if conflict_summary:
            agreement_val = 0.35
        elif tech_sig.classification in ["BULLISH", "STRONG_BULLISH"] and fund_sig.filing_verdict == "POSITIVE":
            agreement_val = 0.92
        else:
            agreement_val = 0.65

        evidence_val = 0.95 if (fund_sig.rag_citations and not is_degraded_scenario) else 0.25
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
            market_view=market_view,
            synthesized_verdict=final_verdict,
            market_classification=market_class,
            confidence=confidence_breakdown.composite_confidence,
            investor_fit_score=investor_fit_score,
            reasoning_trace=reasoning_trace,
            conflict_summary=conflict_summary,
            source_attributions=fund_sig.rag_citations,
            personalized_advice=advice,
            risk_profile=persona,
            degraded_data=degraded_tracker.is_degraded,
            llm_used=llm_used,
            fallback_used=fallback_used,
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
            },
            llm_used=llm_used,
            fallback_used=fallback_used
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
