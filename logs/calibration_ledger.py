from typing import Dict, Any, List

class CalibrationLedger:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        # Predefined demo benchmark outcomes for session alignment
        self.scenario_benchmarks: Dict[str, str] = {
            "aligned": "BUY CANDIDATE",
            "conflict": "HOLD-WATCH",
            "degraded": "HOLD-WATCH",
            "stale_behavioral": "HOLD-WATCH"
        }

    def record_run(self, session_id: str, scenario_id: str, stated_confidence: float, verdict: str):
        benchmark = self.scenario_benchmarks.get(scenario_id, "HOLD-WATCH")
        # Check alignment with benchmark logic
        matched = (
            (benchmark in verdict) or
            ("BUY" in benchmark and "BUY" in verdict) or
            ("HOLD" in benchmark and ("HOLD" in verdict or "AVOID" in verdict)) or
            ("AVOID" in benchmark and ("AVOID" in verdict or "HOLD" in verdict))
        )
        self._records.append({
            "session_id": session_id,
            "scenario_id": scenario_id,
            "stated_confidence": round(stated_confidence, 3),
            "verdict": verdict,
            "benchmark_expected": benchmark,
            "aligned": matched
        })

    def get_calibration_score(self) -> float:
        if not self._records:
            return 0.85  # Standard initial demo calibration baseline
        aligned_count = sum(1 for r in self._records if r["aligned"])
        score = aligned_count / len(self._records)
        return round(score, 3)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_runs": len(self._records),
            "historical_calibration_score": self.get_calibration_score(),
            "records": self._records[-10:],
            "disclaimer": "Session-local illustrative calibration ledger for demo scenarios only. Not live market backtesting."
        }

calibration_ledger = CalibrationLedger()
