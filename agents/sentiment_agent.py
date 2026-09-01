from typing import Dict, Any, List
from schemas import SentimentSignal

class SentimentAgent:
    def __init__(self):
        self.name = "SentimentAgent"

    def analyze(self, ticker: str, market_data: Dict[str, Any], degraded_override: bool = False) -> SentimentSignal:
        """
        Calculates sentiment classification and metrics over news headlines,
        social chatter, and institutional flow trends.
        """
        if degraded_override or not market_data:
            return SentimentSignal(
                ticker=ticker,
                news_sentiment=0.0,
                fii_flow_trend="NEUTRAL",
                social_chatter_score=0.5,
                classification="NEUTRAL",
                confidence=0.30,
                reasons=["Sentiment feed degraded; default neutral posture applied."],
                degraded=True
            )

        headlines = market_data.get("news_headlines", [])
        social_score = market_data.get("social_chatter_score", 0.5)
        fii_flow = market_data.get("fii_flow_trend", "NEUTRAL")

        # Compute average headline sentiment
        if headlines:
            sent_scores = [h.get("sentiment", 0.0) for h in headlines]
            avg_headline_sent = sum(sent_scores) / len(sent_scores)
        else:
            avg_headline_sent = 0.0

        reasons: List[str] = []

        # Evaluate flow score
        flow_weight = 0.0
        if fii_flow == "INFLOW":
            flow_weight = 0.3
            reasons.append("Institutional FII/DII net data reflects continuous accumulation and positive liquidity support.")
        elif fii_flow == "OUTFLOW":
            flow_weight = -0.3
            reasons.append("Institutional FII/DII net figures indicate distribution and capital outflows.")
        else:
            reasons.append("Institutional investment flows remain balanced with no distinct directional bias.")

        # Evaluate headline score
        if avg_headline_sent > 0.6:
            reasons.append(f"Media coverage is strongly favorable (score: +{avg_headline_sent:.2f}) highlighting operational milestones.")
        elif avg_headline_sent > 0.2:
            reasons.append(f"Media narrative is moderately constructive (score: +{avg_headline_sent:.2f}).")
        elif avg_headline_sent < -0.2:
            reasons.append(f"Media coverage displays negative tone (score: {avg_headline_sent:.2f}) focusing on headwinds.")
        else:
            reasons.append("Media mentions remain largely neutral with routine corporate reporting.")

        # Evaluate social chatter
        if social_score > 0.8:
            reasons.append(f"Social chatter intensity is very high ({social_score*100:.0f}%), signaling retail momentum focus.")
        elif social_score < 0.3:
            reasons.append(f"Retail social interest is subdued ({social_score*100:.0f}%).")

        # Composite sentiment calculation
        composite = (avg_headline_sent * 0.5) + flow_weight + ((social_score - 0.5) * 0.4)
        composite = max(-1.0, min(1.0, composite))

        if composite >= 0.45:
            classification = "POSITIVE"
            confidence = min(0.92, 0.68 + (composite * 0.25))
        elif composite >= 0.15:
            classification = "POSITIVE" if avg_headline_sent > 0.2 else "NEUTRAL"
            confidence = 0.72
        elif composite <= -0.40:
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
            fii_flow_trend=fii_flow,
            social_chatter_score=round(social_score, 2),
            classification=classification,
            confidence=round(confidence, 2),
            reasons=reasons,
            degraded=False
        )

sentiment_agent = SentimentAgent()
