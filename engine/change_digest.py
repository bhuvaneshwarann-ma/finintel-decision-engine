from typing import Dict, Any, List, Optional
from schemas import ChangeDigest, ChangeItem, ThesisBreakEvent, BehavioralDriftReport

class ChangeDigestEngine:
    def compute_changes(
        self,
        ticker: str,
        current_data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]],
        since_session_id: Optional[str] = None
    ) -> ChangeDigest:
        """
        Calculates prioritized differences between current analysis and previous session.
        Returns an empty digest when nothing material changed (Test 21).
        """
        if not previous_data:
            return ChangeDigest(ticker=ticker, since_session_id=since_session_id, changed_items=[])

        changes: List[ChangeItem] = []

        curr_payload = current_data.get("raw_result_payload", {})
        prev_payload = previous_data.get("raw_result_payload", {})

        # 1. Thesis Break Changes (Highest Priority)
        curr_breaks: List[Dict[str, Any]] = curr_payload.get("thesis_break_events", [])
        prev_breaks: List[Dict[str, Any]] = prev_payload.get("thesis_break_events", [])
        
        curr_assumptions = {b.get("broken_assumption") for b in curr_breaks if isinstance(b, dict)}
        prev_assumptions = {b.get("broken_assumption") for b in prev_breaks if isinstance(b, dict)}

        new_breaks = curr_assumptions - prev_assumptions
        for b in curr_breaks:
            if isinstance(b, dict) and b.get("broken_assumption") in new_breaks:
                changes.append(ChangeItem(
                    category="THESIS",
                    description=f"New Thesis Break: {b.get('broken_assumption')} - {b.get('explanation')}",
                    citation_tag=b.get("evidence_citation"),
                    materiality="HIGH"
                ))

        # 2. Behavioral Drift Changes
        curr_drift = curr_payload.get("behavioral_drift_report", {})
        prev_drift = prev_payload.get("behavioral_drift_report", {})
        curr_sev = curr_drift.get("drift_severity", "NONE")
        prev_sev = prev_drift.get("drift_severity", "NONE")
        if curr_sev != prev_sev:
            changes.append(ChangeItem(
                category="RISK",
                description=f"Behavioral Drift severity shifted from {prev_sev} to {curr_sev}.",
                citation_tag=None,
                materiality="MEDIUM" if curr_sev == "MILD" else "HIGH"
            ))

        # 3. Market Classification Changes
        curr_verdict = current_data.get("synthesized_verdict")
        prev_verdict = previous_data.get("synthesized_verdict")
        if curr_verdict and prev_verdict and curr_verdict != prev_verdict:
            changes.append(ChangeItem(
                category="MARKET",
                description=f"Decision verdict updated from '{prev_verdict}' to '{curr_verdict}'.",
                citation_tag=None,
                materiality="HIGH"
            ))

        curr_class = current_data.get("signal_classification")
        prev_class = previous_data.get("signal_classification")
        if curr_class and prev_class and curr_class != prev_class:
            changes.append(ChangeItem(
                category="MARKET",
                description=f"Market technical/fundamental stance changed from '{prev_class}' to '{curr_class}'.",
                citation_tag=None,
                materiality="MEDIUM"
            ))

        return ChangeDigest(
            ticker=ticker,
            since_session_id=since_session_id or previous_data.get("session_id"),
            changed_items=changes
        )

change_digest_engine = ChangeDigestEngine()
