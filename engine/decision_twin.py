from typing import List, Optional
from schemas import DecisionTwin, TechnicalSignal, FundamentalSignal, SentimentSignal, RiskAssessment, ThesisBreakEvent, BehavioralDriftReport

class DecisionTwinEngine:
    def compute_twin(
        self,
        ticker: str,
        persona: str,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal,
        risk_assessment: RiskAssessment,
        thesis_breaks: List[ThesisBreakEvent],
        behavioral_drift: BehavioralDriftReport,
        filing_warning: Optional[str] = None
    ) -> DecisionTwin:
        """
        Computes the five separately scored dimensions of the Investment Decision Twin.
        Guarantees scores are independently calculated and never averaged into a single score (Test 17).
        """
        # 1. Market Confidence (Alignment between technical, sentiment, fundamentals)
        # Agreement score between technical and sentiment vs fundamental
        align_score = 0.50
        if (technical_sig.classification in ["BULLISH", "STRONG_BULLISH"]) and (sentiment_sig.classification == "POSITIVE"):
            align_score = 0.85 if fundamental_sig.filing_verdict == "POSITIVE" else 0.45
        elif (technical_sig.classification in ["BEARISH", "STRONG_BEARISH"]) and (sentiment_sig.classification == "NEGATIVE"):
            align_score = 0.85 if fundamental_sig.filing_verdict in ["CONCERNING", "CRITICAL_RISK"] else 0.50
        else:
            align_score = 0.65

        if technical_sig.degraded or fundamental_sig.degraded or sentiment_sig.degraded:
            align_score = max(0.30, align_score - 0.25)
        market_conf = round(float(align_score), 2)

        # 2. Decision Fit (Suitability for persona constraints, concentration, debt penalties)
        # Conservative strongly penalizes debt & high risk score; Aggressive tolerates higher momentum
        if persona == "conservative":
            fit_score = 1.0 - risk_assessment.risk_score
            if fundamental_sig.debt_to_equity > 1.2:
                fit_score = max(0.15, fit_score - 0.40)
        else:
            fit_score = 0.90 if technical_sig.classification in ["STRONG_BULLISH", "BULLISH"] else 0.60
            if fundamental_sig.debt_to_equity > 3.0:
                fit_score = max(0.30, fit_score - 0.30)
        decision_fit = round(float(fit_score), 2)

        # 3. Evidence Quality (Freshness and citation density)
        if filing_warning:
            ev_quality = 0.40  # Stale evidence
        elif fundamental_sig.degraded:
            ev_quality = 0.20  # Missing evidence
        elif len(fundamental_sig.rag_citations) >= 2:
            ev_quality = 0.95
        elif len(fundamental_sig.rag_citations) == 1:
            ev_quality = 0.85
        else:
            ev_quality = 0.50
        evidence_quality = round(float(ev_quality), 2)

        # 4. Thesis Health (Assumptions validity)
        if not thesis_breaks:
            thesis_health = 0.92
        elif any(tb.severity == "BROKEN" for tb in thesis_breaks):
            thesis_health = 0.20
        elif any(tb.severity == "WEAKENED" for tb in thesis_breaks):
            thesis_health = 0.55
        else:
            thesis_health = 0.75
        thesis_health = round(float(thesis_health), 2)

        # 5. Behavioral Risk (Bias and drift score; 1.0 = high risk/FOMO, 0.0 = low risk)
        if behavioral_drift.drift_severity == "NOTABLE":
            b_risk = 0.85
        elif behavioral_drift.drift_severity == "MILD":
            b_risk = 0.50
        elif len(risk_assessment.behavioral_flags) > 0:
            b_risk = 0.45
        else:
            b_risk = 0.10
        behavioral_risk = round(float(b_risk), 2)

        # Generate composite note explaining divergence without averaging
        notes = []
        if abs(market_conf - decision_fit) >= 0.25:
            notes.append(f"Market confidence ({market_conf}) diverges from investor decision fit ({decision_fit}) due to portfolio/debt constraints.")
        if thesis_health <= 0.4:
            notes.append(f"Core thesis assumptions are impaired (health: {thesis_health}).")
        if behavioral_risk >= 0.5:
            notes.append(f"Behavioral drift elevated ({behavioral_risk}); potential FOMO or concentration chasing.")
        if not notes:
            notes.append("All five decision dimensions reflect consistent positioning with no acute anomalies.")

        composite_note = " ".join(notes)

        return DecisionTwin(
            ticker=ticker,
            persona=persona,
            market_confidence=market_conf,
            decision_fit=decision_fit,
            evidence_quality=evidence_quality,
            thesis_health=thesis_health,
            behavioral_risk=behavioral_risk,
            composite_note=composite_note
        )

decision_twin_engine = DecisionTwinEngine()
