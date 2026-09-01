from typing import Dict, Any, List
from schemas import SentimentSignal

class SentimentAgent:
    def __init__(self):
        self.name = "SentimentAgent"

    def analyze(self, ticker: str, market_data: Dict[str, Any], degraded_override: bool = False) -> SentimentSignal:
        """
        Calculates sentiment classification and metrics over news headlines,
        social chatter, and separate FII/DII institutional flow trends.
        """
        if degraded_override or not market_data:
            return SentimentSignal(
                ticker=ticker,
                news_sentiment=0.0,
                fii_flow="NEUTRAL",
                dii_flow="NEUTRAL",
                fii_flow_trend="NEUTRAL",
                social_chatter_score=0.5,
                classification="NEUTRAL",
                confidence=0.30,
                reasons=["Sentiment feed degraded; default neutral posture applied."],
                degraded=True
            )

        headlines = market_data.get("news_headlines", [])
        social_score = market_data.get("social_chatter_score", 0.5)
        fii_flow = market_data.get("fii_flow", market_data.get("fii_flow_trend", "NEUTRAL"))
        dii_flow = market_data.get("dii_flow", "NEUTRAL")

        # Compute average headline sentiment
        if headlines:
            sent_scores = [h.get("sentiment", 0.0) for h in headlines]
            avg_headline_sent = sum(sent_scores) / len(sent_scores)
        else:
            avg_headline_sent = 0.0

        reasons: List[str] = []

        # 1. Evaluate FII / DII flows separately (§10)
        flow_weight = 0.0
        if fii_flow == "INFLOW" and dii_flow == "INFLOW":
            flow_weight = 0.35
            reasons.append("Coordinated Institutional Accumulation: Both FII (+Inflow) and DII (+Inflow) reflect positive liquidity inflows.")
        elif fii_flow == "INFLOW":
            flow_weight = 0.25
            reasons.append(f"Foreign Institutional Investors (FII) demonstrate net accumulation (+Inflow), while Domestic Institutions (DII) are {dii_flow}.")
        elif dii_flow == "INFLOW":
            flow_weight = 0.20
            reasons.append(f"Domestic Institutions (DII) show net buying (+Inflow), while Foreign Institutions (FII) are {fii_flow}.")
        elif fii_flow == "OUTFLOW" or dii_flow == "OUTFLOW":
            flow_weight = -0.30
            reasons.append(f"Institutional distribution observed (FII: {fii_flow}, DII: {dii_flow}).")
        else:
            reasons.append("Institutional flows (FII and DII) remain balanced without strong directional bias.")

        # 2. Evaluate headline score
        if avg_headline_sent > 0.6:
            reasons.append(f"Media narrative is strongly constructive (+{avg_headline_sent:.2f}) highlighting operational milestones.")
        elif avg_headline_sent > 0.2:
            reasons.append(f"Media coverage is moderately positive (+{avg_headline_sent:.2f}).")
        elif avg_headline_sent < -0.2:
            reasons.append(f"Media coverage displays negative tone ({avg_headline_sent:.2f}) focusing on balance sheet or operational headwinds.")
        else:
            reasons.append("Media mentions remain largely neutral with routine corporate reporting.")

        # 3. Evaluate social chatter
        if social_score > 0.8:
            reasons.append(f"Social media chatter intensity is elevated ({social_score*100:.0f}%), signaling active retail participation.")
        elif social_score < 0.3:
            reasons.append(f"Retail social volume is subdued ({social_score*100:.0f}%).")

        # Composite sentiment calculation
        composite = (avg_headline_sent * 0.45) + flow_weight + ((social_score - 0.5) * 0.35)
        composite = max(-1.0, min(1.0, composite))

        if composite >= 0.40:
            classification = "POSITIVE"
            confidence = min(0.92, 0.70 + (composite * 0.22))
        elif composite >= 0.15:
            classification = "POSITIVE" if avg_headline_sent > 0.2 else "NEUTRAL"
            confidence = 0.72
        elif composite <= -0.35:
            classification = "NEGATIVE"
            confidence = min(0.90, 0.70 + (abs(composite) * 0.20))
        elif composite <= -0.15:
            classification = "NEGATIVE" if avg_headline_sent < -0.2 else "NEUTRAL"
            confidence = 0.68
        else:
            classification = "NEUTRAL"
            confidence = 0.60

        return SentimentSignal(
            ticker=ticker,
            news_sentiment=round(avg_headline_sent, 3),
            fii_flow=fii_flow,
            dii_flow=dii_flow,
            fii_flow_trend=fii_flow,
            social_chatter_score=round(social_score, 2),
            classification=classification,
            confidence=round(confidence, 2),
            reasons=reasons,
            degraded=False
        )

sentiment_agent = SentimentAgent()
