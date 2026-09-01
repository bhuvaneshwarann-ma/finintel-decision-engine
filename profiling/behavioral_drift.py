from typing import Dict, Any, List
from schemas import BehavioralDriftReport, BehavioralFlag
from profiling.behavioral_bias_mirror import detect_behavioral_biases
from utils.security import check_no_diagnostic_terms

def evaluate_behavioral_drift(
    persona_id: str,
    ticker: str,
    portfolio: Dict[str, float],
    market_data: Dict[str, Any],
    scenario: str = "aligned"
) -> BehavioralDriftReport:
    """
    Evaluates behavioral drift by comparing declared persona traits against
    observed simulated interaction patterns.
    Guarantees no diagnostic/clinical labels are used (Test 22).
    """
    flags = detect_behavioral_biases(persona_id, ticker, portfolio, market_data, scenario)
    user_actions = market_data.get("simulated_user_actions", {})
    req_count = user_actions.get("request_count_past_hour", 0)
    size_mult = user_actions.get("requested_position_size_multiplier", 1.0)
    spike_pct = user_actions.get("recent_price_spike_pct", 0.0)

    if not flags:
        severity = "NONE"
        pattern = f"Observed interaction pattern aligns with declared {persona_id} risk parameters. Orderly inquiry rate ({req_count}/hr)."
    elif len(flags) == 1:
        severity = "MILD"
        pattern = f"Mild variance detected: {req_count} queries in past hour with {size_mult}x position sizing interest following {spike_pct}% movement."
    else:
        severity = "NOTABLE"
        pattern = f"Notable variance detected: Elevated inquiry frequency ({req_count}/hr) and aggressive sizing ({size_mult}x) conflicting with baseline risk posture."

    # Validate that output contains no diagnostic terms
    assert check_no_diagnostic_terms(pattern), "Diagnostic terms detected in pattern description"
    for flag in flags:
        assert check_no_diagnostic_terms(flag.nudge_message), "Diagnostic terms detected in nudge message"

    return BehavioralDriftReport(
        declared_risk_profile=persona_id,
        observed_behavior_pattern=pattern,
        drift_flags=flags,
        drift_severity=severity
    )
