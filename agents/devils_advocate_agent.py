from typing import Dict, Any, List, Optional
from schemas import DevilsAdvocateChallenge, TechnicalSignal, FundamentalSignal, SentimentSignal, ThesisRecord

class DevilsAdvocateAgent:
    def __init__(self):
        self.name = "DevilsAdvocateAgent"

    def challenge(
        self,
        ticker: str,
        draft_verdict: str,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal,
        existing_citations: List[str],
        thesis_record: Optional[ThesisRecord] = None
    ) -> DevilsAdvocateChallenge:
        """
        Adversarial challenge pass that stress-tests the draft verdict.
        Strictly constrained to only re-weigh existing evidence and citations.
        Never invents new citations or facts.
        """
        counter_arg = ""
        weakest_point = ""
        severity = "LOW"
        conf_adj = 0.0

        # Case 1: Draft verdict is Bullish/Buy, but fundamental or leverage risk exists
        if "BUY" in draft_verdict or "BULLISH" in draft_verdict:
            if fundamental_sig.debt_to_equity is not None and fundamental_sig.debt_to_equity > 2.0:
                severity = "HIGH"
                conf_adj = -0.18
                citation_ref = existing_citations[0] if existing_citations else "regulatory filings"
                counter_arg = (
                    f"The draft bullish verdict relies heavily on short-term price and sentiment momentum, "
                    f"while discounting severe balance sheet leverage (Debt/Equity: {fundamental_sig.debt_to_equity:.2f}x) "
                    f"disclosed in {citation_ref}. In a liquidity crunch, equity value could face severe dilution."
                )
                weakest_point = f"Reliance on technical breakout over disclosed debt distress ({citation_ref})."
            elif fundamental_sig.degraded or not fundamental_sig.data_available:
                severity = "MEDIUM"
                conf_adj = -0.12
                counter_arg = (
                    "The draft verdict assumes ongoing fundamental health, but the underlying regulatory filing "
                    "evidence is stale or degraded. Solvency cannot be verified from outdated disclosures."
                )
                weakest_point = "Unverified fundamental health due to stale/degraded filing disclosures."
            elif technical_sig.rsi > 70:
                severity = "MEDIUM"
                conf_adj = -0.08
                counter_arg = (
                    f"RSI ({technical_sig.rsi}) is in extended overbought territory. "
                    f"A momentum reversal could trigger swift drawdowns despite positive sentiment."
                )
                weakest_point = f"Overbought technical stretch (RSI {technical_sig.rsi})."
            else:
                severity = "LOW"
                conf_adj = -0.02
                counter_arg = (
                    "While signals appear aligned, execution risks around quarterly revenue guidance and "
                    "institutional profit-taking should be monitored."
                )
                weakest_point = "Assumes sustained historical margins without macroeconomic slowdown."

        # Case 2: Draft verdict is Bearish/Avoid, but strong momentum or healthy cash reserves exist
        elif "AVOID" in draft_verdict or "BEARISH" in draft_verdict or "REDUCE" in draft_verdict:
            if technical_sig.classification in ["STRONG_BULLISH", "BULLISH"] and sentiment_sig.classification == "POSITIVE":
                severity = "MEDIUM"
                conf_adj = -0.05
                counter_arg = (
                    f"The defensive verdict may cause an investor to miss strong institutional breakout momentum "
                    f"(Volume Z: +{technical_sig.volume_anomaly_score:.2f}, FII Flow: {sentiment_sig.fii_flow})."
                )
                weakest_point = "Underestimating short-term liquidity momentum and retail sentiment."
            else:
                severity = "LOW"
                counter_arg = (
                    "Defensive caution is well-supported by evidence, though severe undervaluation could emerge if market overreacts."
                )
                weakest_point = "Assumes downside risks outweigh potential valuation stabilization."

        # Case 3: Draft verdict is Hold/Watch
        else:
            severity = "LOW"
            counter_arg = (
                "A hold-watch recommendation is prudent under mixed signals, though opportunity cost exists if breakout sustains."
            )
            weakest_point = "Potential delay in capitalizing on early stage recovery."

        return DevilsAdvocateChallenge(
            ticker=ticker,
            target_verdict=draft_verdict,
            counter_argument=counter_arg,
            weakest_evidence_point=weakest_point,
            severity=severity,
            confidence_adjustment=conf_adj
        )

devils_advocate_agent = DevilsAdvocateAgent()
