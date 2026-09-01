from schemas import ConfidenceBreakdown

# Weights for confidence decomposition (sum to 1.0)
WEIGHT_FRESHNESS = 0.25
WEIGHT_AGREEMENT = 0.35
WEIGHT_EVIDENCE = 0.25
WEIGHT_CALIBRATION = 0.15

def clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, val))

def compute_confidence_breakdown(
    data_freshness: float,
    agent_agreement: float,
    evidence_strength: float,
    historical_calibration: float
) -> ConfidenceBreakdown:
    """
    Computes a mathematical ConfidenceBreakdown ensuring all scores are bounded [0, 1]
    and composite_confidence equals the weighted sum.
    """
    f = clamp(round(data_freshness, 3))
    a = clamp(round(agent_agreement, 3))
    e = clamp(round(evidence_strength, 3))
    c = clamp(round(historical_calibration, 3))
    
    composite = round(
        (f * WEIGHT_FRESHNESS) +
        (a * WEIGHT_AGREEMENT) +
        (e * WEIGHT_EVIDENCE) +
        (c * WEIGHT_CALIBRATION),
        3
    )
    composite = clamp(composite)
    
    return ConfidenceBreakdown(
        data_freshness_score=f,
        agent_agreement_score=a,
        evidence_strength_score=e,
        historical_calibration_score=c,
        composite_confidence=composite
    )
