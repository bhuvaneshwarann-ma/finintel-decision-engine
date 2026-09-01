from typing import List, Dict, Any
from schemas import ScenarioSimulation, TechnicalSignal, FundamentalSignal, SentimentSignal
from profiling.user_profile import get_persona_profile

DISCLAIMER_TEXT = "Simulated illustration based on demo data. Not a forecast or guarantee."

class CounterfactualSimulator:
    def __init__(self):
        self.name = "CounterfactualSimulator"

    def simulate(
        self,
        ticker: str,
        persona_id: str,
        technical_sig: TechnicalSignal,
        fundamental_sig: FundamentalSignal,
        sentiment_sig: SentimentSignal,
        thesis_broken_assumption: str = ""
    ) -> List[ScenarioSimulation]:
        """
        Deterministically models Best, Base, Failure, and Thesis-Break scenarios.
        Guarantees simulation_disclaimer is always populated on all scenarios (Test 19).
        """
        profile = get_persona_profile(persona_id)
        current_alloc = profile.get("portfolio", {}).get(ticker, 0.05)

        results: List[ScenarioSimulation] = []

        # 1. Best Scenario
        best_price_move = 0.25 if technical_sig.classification in ["STRONG_BULLISH", "BULLISH"] else 0.12
        impact_best = round(current_alloc * best_price_move * 100, 2)
        conc_best = round((current_alloc * (1 + best_price_move)) / (1 + (current_alloc * best_price_move)), 3)
        results.append(ScenarioSimulation(
            ticker=ticker,
            scenario_type="BEST",
            assumption_changed="Technical momentum and revenue expansion persist at upper historical bounds (+25%).",
            projected_portfolio_impact_pct=impact_best,
            projected_concentration_after=conc_best,
            narrative_explanation=f"If operational execution exceeds targets, portfolio gains approx +{impact_best}% with position concentration shifting to {conc_best*100:.1f}%.",
            simulation_disclaimer=DISCLAIMER_TEXT
        ))

        # 2. Base Scenario
        base_price_move = 0.04
        impact_base = round(current_alloc * base_price_move * 100, 2)
        conc_base = round(current_alloc, 3)
        results.append(ScenarioSimulation(
            ticker=ticker,
            scenario_type="BASE",
            assumption_changed="Current valuation multiples and earnings trajectory remain stable (+4%).",
            projected_portfolio_impact_pct=impact_base,
            projected_concentration_after=conc_base,
            narrative_explanation=f"In a steady-state scenario, position yields +{impact_base}% portfolio contribution with stable concentration at {conc_base*100:.1f}%.",
            simulation_disclaimer=DISCLAIMER_TEXT
        ))

        # 3. Failure Scenario
        fail_assumption = "Macro slowdown or quarterly earnings miss triggers a 20% valuation compression."
        fail_price_move = -0.20
        impact_fail = round(current_alloc * fail_price_move * 100, 2)
        conc_fail = round((current_alloc * (1 + fail_price_move)) / (1 + (current_alloc * fail_price_move)), 3)
        results.append(ScenarioSimulation(
            ticker=ticker,
            scenario_type="FAILURE",
            assumption_changed=fail_assumption,
            projected_portfolio_impact_pct=impact_fail,
            projected_concentration_after=conc_fail,
            narrative_explanation=f"A standard adverse repricing produces a {impact_fail}% portfolio drag; concentration falls to {conc_fail*100:.1f}%.",
            simulation_disclaimer=DISCLAIMER_TEXT
        ))

        # 4. Thesis-Break Scenario
        tb_assumption = (
            thesis_broken_assumption if thesis_broken_assumption 
            else f"Debt distress / covenant breach materializes (Debt/Equity: {fundamental_sig.debt_to_equity:.2f}x)."
        )
        tb_price_move = -0.45 if fundamental_sig.debt_to_equity > 2.0 else -0.30
        impact_tb = round(current_alloc * tb_price_move * 100, 2)
        conc_tb = round((current_alloc * (1 + tb_price_move)) / (1 + (current_alloc * tb_price_move)), 3)
        results.append(ScenarioSimulation(
            ticker=ticker,
            scenario_type="THESIS_BREAK",
            assumption_changed=f"Thesis Invalidation: {tb_assumption}",
            projected_portfolio_impact_pct=impact_tb,
            projected_concentration_after=conc_tb,
            narrative_explanation=f"If the stated thesis is invalidated by fundamental deterioration, severe downside risk (-45%) yields a {impact_tb}% portfolio hit.",
            simulation_disclaimer=DISCLAIMER_TEXT
        ))

        return results

counterfactual_simulator = CounterfactualSimulator()
