from typing import List, Dict, Any
from schemas import ScenarioSimulation, TechnicalSignal, FundamentalSignal, SentimentSignal
from profiling.user_profile import get_persona_profile

class CounterfactualSimulator:
    def __init__(self):
        self.name = "RiskScenarioSimulator"

    def simulate(
        self,
        ticker: str,
        persona_id: str,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal,
        multiplier: float = 1.0
    ) -> List[ScenarioSimulation]:
        """
        Generates deterministic risk scenario impact analysis across 4 defined states:
        Best Case, Base Case, Failure Case, and Thesis-Break State.
        """
        profile = get_persona_profile(persona_id)
        portfolio = profile.get("portfolio", {})
        base_alloc = portfolio.get(ticker, 0.05)
        effective_alloc = min(0.50, base_alloc * multiplier)

        scenarios: List[ScenarioSimulation] = []

        # 1. Best Case Scenario (Upside Momentum Continuation)
        upside_return = 0.25
        best_impact = effective_alloc * upside_return * 100
        scenarios.append(
            ScenarioSimulation(
                ticker=ticker,
                scenario_type="BEST",
                assumption_changed="Bullish technical momentum & revenue catalysts sustain at upper forecast bound (+25%).",
                projected_portfolio_impact_pct=round(best_impact, 2),
                projected_concentration_after=round(effective_alloc * 1.25, 3),
                narrative_explanation=(
                    f"If favorable liquidity and earnings continuation hold, an effective allocation of "
                    f"{effective_alloc*100:.1f}% contributes approximately +{best_impact:.2f}% to total portfolio return."
                ),
                simulation_disclaimer="Illustrative educational scenario based on demo parameters. Not a price forecast or guarantee."
            )
        )

        # 2. Base Case Scenario (Steady Multiples)
        base_return = 0.04
        base_impact = effective_alloc * base_return * 100
        scenarios.append(
            ScenarioSimulation(
                ticker=ticker,
                scenario_type="BASE",
                assumption_changed="Current valuation multiples and operating margins remain stable (+4%).",
                projected_portfolio_impact_pct=round(base_impact, 2),
                projected_concentration_after=round(effective_alloc * 1.04, 3),
                narrative_explanation=(
                    f"Under neutral macro and baseline operating delivery, the simulated position provides "
                    f"+{base_impact:.2f}% portfolio contribution."
                ),
                simulation_disclaimer="Illustrative educational scenario based on demo parameters. Not a price forecast or guarantee."
            )
        )

        # 3. Failure Case Scenario (Cyclical Headwind / Earnings Miss)
        downside_return = -0.20
        fail_impact = effective_alloc * downside_return * 100
        scenarios.append(
            ScenarioSimulation(
                ticker=ticker,
                scenario_type="FAILURE",
                assumption_changed="Sectoral slowdown or earnings miss induces a cyclical multiple compression (-20%).",
                projected_portfolio_impact_pct=round(fail_impact, 2),
                projected_concentration_after=round(effective_alloc * 0.80, 3),
                narrative_explanation=(
                    f"A standard adverse cyclical correction reduces position value, generating a "
                    f"{fail_impact:.2f}% portfolio drawdown at {multiplier:.1f}x sizing."
                ),
                simulation_disclaimer="Illustrative educational scenario based on demo parameters. Not a price forecast or guarantee."
            )
        )

        # 4. Thesis-Break Case Scenario (Severe Leverage Distress Materializes)
        thesis_break_return = -0.45
        tb_impact = effective_alloc * thesis_break_return * 100
        scenarios.append(
            ScenarioSimulation(
                ticker=ticker,
                scenario_type="THESIS_BREAK",
                assumption_changed="Debt refinancing distress or core covenant breach materializes fully (-45%).",
                projected_portfolio_impact_pct=round(tb_impact, 2),
                projected_concentration_after=round(effective_alloc * 0.55, 3),
                narrative_explanation=(
                    f"Severe fundamental stress triggers capital impairment, creating a "
                    f"{tb_impact:.2f}% portfolio loss at current sizing."
                ),
                simulation_disclaimer="Illustrative educational scenario based on demo parameters. Not a price forecast or guarantee."
            )
        )

        return scenarios

counterfactual_simulator = CounterfactualSimulator()
