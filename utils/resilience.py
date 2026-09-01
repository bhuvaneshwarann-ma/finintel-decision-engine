from typing import Dict, Any, List

class DegradedTracker:
    def __init__(self):
        self.is_degraded: bool = False
        self.degraded_components: List[str] = []
        self.fallbacks_used: List[str] = []
        self.reasons: List[str] = []
        self.confidence_penalty: float = 0.0

    def mark_degraded(self, component: str, reason: str, fallback_used: str = "", penalty: float = 0.2):
        self.is_degraded = True
        if component not in self.degraded_components:
            self.degraded_components.append(component)
        if fallback_used and fallback_used not in self.fallbacks_used:
            self.fallbacks_used.append(fallback_used)
        self.reasons.append(f"[{component}] {reason}")
        self.confidence_penalty = min(0.6, self.confidence_penalty + penalty)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "degraded_data": self.is_degraded,
            "degraded_components": self.degraded_components,
            "fallbacks_used": self.fallbacks_used,
            "reasons": self.reasons,
            "confidence_penalty": round(self.confidence_penalty, 2)
        }
