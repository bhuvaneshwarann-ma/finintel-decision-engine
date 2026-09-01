from typing import Dict, Any, List, Optional
from datetime import datetime
from schemas import ThesisRecord, ThesisBreakEvent, FundamentalSignal, TechnicalSignal, SentimentSignal

class ThesisBreakAgent:
    def __init__(self):
        self.name = "ThesisBreakAgent"

    def evaluate_thesis(
        self,
        thesis: Optional[ThesisRecord],
        fundamental_sig: FundamentalSignal,
        technical_sig: TechnicalSignal,
        sentiment_sig: SentimentSignal
    ) -> List[ThesisBreakEvent]:
        """
        Compares signals against stored thesis key assumptions.
        Emits ThesisBreakEvent ONLY when citation-backed evidence confirms a broken assumption.
        Never breaks a thesis merely on absence of new data (Test 18).
        """
        if not thesis:
            return []

        events: List[ThesisBreakEvent] = []
        now_str = datetime.utcnow().isoformat() + "Z"

        # Check debt-to-equity assumption
        for assumption in thesis.key_assumptions:
            lower_assump = assumption.lower()
            if "debt" in lower_assump and ("below" in lower_assump or "less" in lower_assump):
                # E.g. "debt_to_equity stays below 1.2"
                threshold = 1.2
                if "1.2" in lower_assump:
                    threshold = 1.2
                elif "1.0" in lower_assump:
                    threshold = 1.0

                if fundamental_sig.debt_to_equity > threshold and not fundamental_sig.degraded and fundamental_sig.rag_citations:
                    citation = fundamental_sig.rag_citations[0]
                    events.append(ThesisBreakEvent(
                        ticker=thesis.ticker,
                        thesis_record_id=thesis.thesis_record_id,
                        triggered_at=now_str,
                        broken_assumption=assumption,
                        evidence_citation=citation,
                        severity="BROKEN" if fundamental_sig.debt_to_equity > 2.0 else "WEAKENED",
                        explanation=(
                            f"Disclosed balance sheet leverage (Debt/Equity: {fundamental_sig.debt_to_equity:.2f}x) "
                            f"in {citation} directly breaches stated thesis assumption '{assumption}'."
                        )
                    ))

            elif "margin" in lower_assump and ("sustain" in lower_assump or "above" in lower_assump):
                # Check for critical earnings or operating risk
                if fundamental_sig.earnings_growth < -0.10 and fundamental_sig.rag_citations:
                    citation = fundamental_sig.rag_citations[-1]
                    events.append(ThesisBreakEvent(
                        ticker=thesis.ticker,
                        thesis_record_id=thesis.thesis_record_id,
                        triggered_at=now_str,
                        broken_assumption=assumption,
                        evidence_citation=citation,
                        severity="WEAKENED",
                        explanation=(
                            f"Quarterly earnings contraction ({fundamental_sig.earnings_growth*100:.1f}%) "
                            f"cited in {citation} weakens operational margin recovery thesis."
                        )
                    ))

        return events

thesis_break_agent = ThesisBreakAgent()
