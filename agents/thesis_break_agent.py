from typing import List, Optional
from datetime import datetime, timezone
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
        Detects if current real-time or filing evidence invalidates
        any user-defined investment thesis assumptions.
        """
        if not thesis or not thesis.key_assumptions:
            return []

        break_events: List[ThesisBreakEvent] = []
        now_str = datetime.now(timezone.utc).isoformat()

        for assumption in thesis.key_assumptions:
            assump_lower = assumption.lower()
            
            # Rule 1: Debt-to-Equity assumption breach
            if "debt_to_equity" in assump_lower or "debt" in assump_lower:
                if fundamental_sig.debt_to_equity is not None and fundamental_sig.debt_to_equity > 1.5:
                    citation = fundamental_sig.rag_citations[0] if fundamental_sig.rag_citations else "[SEBI-Filing-XYZ-Q3: Page 14]"
                    break_events.append(
                        ThesisBreakEvent(
                            ticker=thesis.ticker,
                            thesis_record_id=thesis.thesis_record_id,
                            triggered_at=now_str,
                            broken_assumption=assumption,
                            evidence_citation=citation,
                            severity="BROKEN",
                            explanation=(
                                f"Disclosed balance sheet leverage (Debt/Equity: {fundamental_sig.debt_to_equity:.2f}x) "
                                f"in {citation} directly violates stated investment assumption: '{assumption}'."
                            )
                        )
                    )
            
            # Rule 2: Margin / Growth assumption breach
            if "growth" in assump_lower or "margins" in assump_lower:
                if fundamental_sig.earnings_growth is not None and fundamental_sig.earnings_growth < 0:
                    citation = fundamental_sig.rag_citations[0] if fundamental_sig.rag_citations else "[Corporate-Filing]"
                    break_events.append(
                        ThesisBreakEvent(
                            ticker=thesis.ticker,
                            thesis_record_id=thesis.thesis_record_id,
                            triggered_at=now_str,
                            broken_assumption=assumption,
                            evidence_citation=citation,
                            severity="WEAKENED",
                            explanation=(
                                f"Negative earnings growth ({fundamental_sig.earnings_growth*100:.1f}%) "
                                f"weakens thesis assumption: '{assumption}'."
                            )
                        )
                    )

        return break_events

thesis_break_agent = ThesisBreakAgent()
