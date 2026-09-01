from typing import Dict, Any, List
from schemas import RiskAssessment, TechnicalSignal, FundamentalSignal, SentimentSignal, BehavioralFlag
from profiling.user_profile import get_persona_profile
from profiling.behavioral_bias_mirror import detect_behavioral_biases

class RiskAgent:
    def __init__(self):
        self.name = "RiskAgent"

    def assess_risk(
        self,
        ticker: str,
        persona_name: str,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal,
        market_data: Dict[str, Any],
        scenario: str = "aligned"
    ) -> RiskAssessment:
        """
        Evaluates portfolio concentration, sector risk, balance sheet constraints,
        and behavioral biases for the given investor persona.
        """
        profile = get_persona_profile(persona_name)
        portfolio = profile.get("portfolio", {})
        current_alloc = portfolio.get(ticker, 0.0)
        
        # Sector calculation
        sector = market_data.get("sector", "General")
        sector_exposure = current_alloc  # In bundled 3-stock portfolio, single stock maps to its sector

        max_stock_limit = profile.get("max_single_stock_concentration", 0.15)
        max_sector_limit = profile.get("max_sector_concentration", 0.25)
        debt_tolerance = profile.get("debt_tolerance_ratio", 1.2)

        risk_flags: List[str] = []
        constraints: List[str] = []
        risk_penalties: float = 0.0

        # 1. Evaluate Concentration Limits
        if current_alloc >= max_stock_limit:
            risk_flags.append(f"Position concentration in {ticker} ({current_alloc*100:.1f}%) reaches persona ceiling ({max_stock_limit*100:.1f}%).")
            constraints.append("No fresh capital addition until position is rebalanced.")
            risk_penalties += 0.25
        else:
            constraints.append(f"Allowed incremental sizing up to {round((max_stock_limit - current_alloc)*100, 1)}% portfolio weight.")

        # 2. Evaluate Debt & Fundamental Constraints
        if fundamental_sig.debt_to_equity > debt_tolerance:
            risk_flags.append(
                f"Debt-to-equity ratio ({fundamental_sig.debt_to_equity:.2f}x) breaches {profile['name']}'s risk tolerance threshold ({debt_tolerance:.2f}x)."
            )
            if profile["persona_id"] == "conservative":
                constraints.append("Mandatory veto on aggressive or unhedged exposure due to balance sheet distress.")
                risk_penalties += 0.45
            else:
                constraints.append("High volatility risk flagged; require strict trailing stop-loss.")
                risk_penalties += 0.20

        # 3. Technical & Volatility Risk
        if technical_sig.rsi > 75:
            risk_flags.append("Elevated technical stretch: RSI indicates extended overbought conditions.")
            risk_penalties += 0.15

        # 4. Behavioral Biases
        behavioral_flags = detect_behavioral_biases(
            persona_id=profile["persona_id"],
            ticker=ticker,
            portfolio=portfolio,
            market_data=market_data,
            scenario=scenario
        )
        if behavioral_flags:
            risk_penalties += 0.10 * len(behavioral_flags)

        # Baseline risk score based on persona and penalties
        base_risk = 0.30 if profile["persona_id"] == "conservative" else 0.50
        final_risk_score = min(1.0, base_risk + risk_penalties)

        return RiskAssessment(
            risk_profile=profile["persona_id"],
            portfolio_concentration=current_alloc,
            sector_exposure=sector_exposure,
            risk_flags=risk_flags,
            risk_score=round(final_risk_score, 2),
            personalized_constraints=constraints,
            behavioral_flags=behavioral_flags
        )

risk_agent = RiskAgent()
