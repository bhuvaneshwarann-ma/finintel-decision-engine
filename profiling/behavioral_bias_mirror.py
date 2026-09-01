from typing import List, Dict, Any
from schemas import BehavioralFlag

def detect_behavioral_biases(
    persona_id: str,
    ticker: str,
    portfolio: Dict[str, float],
    market_data: Dict[str, Any],
    scenario: str = "aligned"
) -> List[BehavioralFlag]:
    """
    Evaluates persona, portfolio state, and market context to detect cognitive biases.
    Produces zero flags in control cases where no triggering condition is met.
    """
    flags: List[BehavioralFlag] = []
    user_actions = market_data.get("simulated_user_actions", {})
    recent_spike = user_actions.get("recent_price_spike_pct", 0.0)
    req_count = user_actions.get("request_count_past_hour", 0)
    size_mult = user_actions.get("requested_position_size_multiplier", 1.0)
    current_allocation = portfolio.get(ticker, 0.0)

    # 1. MOMENTUM_CHASING
    # Triggered by sharp price spike (>10%) combined with high query intensity or scenario == 'stale_behavioral'
    if (recent_spike > 10.0 and req_count >= 2) or scenario == "stale_behavioral":
        flags.append(BehavioralFlag(
            flag_type="MOMENTUM_CHASING",
            trigger_reason=f"Multiple rapid inquiries ({req_count} queries) following a recent {recent_spike}% price surge in {ticker}.",
            nudge_message=(
                "Observation: Rapid entry decisions following sharp price spikes often correlate with retail FOMO. "
                "Consider whether your entry is based on fundamental valuation or short-term price excitement."
            )
        ))

    # 2. FAMILIARITY_CONCENTRATION
    # Triggered if already holding heavy exposure (>15% in conservative or >25% in aggressive)
    concentration_threshold = 0.12 if persona_id == "conservative" else 0.25
    if current_allocation >= concentration_threshold and size_mult > 1.0:
        flags.append(BehavioralFlag(
            flag_type="FAMILIARITY_CONCENTRATION",
            trigger_reason=f"Current portfolio allocation in {ticker} is {current_allocation*100:.1f}%, which meets or exceeds recommended single-stock boundaries.",
            nudge_message=(
                "Observation: Adding capital to already heavily weighted positions increases portfolio idiosyncratic risk. "
                "Ensure your overall sector diversification goals are preserved."
            )
        ))

    # 3. LOSS_AVERSION_HOLD
    # Triggered if holding a distressed stock (high debt) and unwilling to trim
    financials = market_data.get("financials", {})
    de_ratio = financials.get("debt_to_equity", 0.0)
    if current_allocation > 0.0 and de_ratio > 3.0 and persona_id == "conservative":
        flags.append(BehavioralFlag(
            flag_type="LOSS_AVERSION_HOLD",
            trigger_reason=f"Holding existing allocation ({current_allocation*100:.1f}%) in {ticker} despite deteriorating debt-to-equity ratio ({de_ratio:.2f}x).",
            nudge_message=(
                "Observation: Investors often hold underperforming positions to avoid realizing a loss. "
                "Evaluate whether you would allocate fresh capital to this business today at current fundamentals."
            )
        ))

    return flags
