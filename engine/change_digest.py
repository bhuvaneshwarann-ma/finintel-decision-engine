from typing import Dict, Any, List, Optional, Union
from schemas import ChangeDigest, ChangeItem, SynthesizedOutput

class ChangeDigestEngine:
    def compute_changes(
        self,
        ticker: str,
        current_data: Union[Dict[str, Any], SynthesizedOutput],
        previous_data: Optional[Union[Dict[str, Any], SynthesizedOutput]],
        since_session_id: Optional[str] = None
    ) -> ChangeDigest:
        """
        Calculates prioritized differences between current analysis and previous session.
        Returns an empty digest when nothing material changed (Test 21).
        """
        session_id = since_session_id or "sess_change_digest"
        if not previous_data:
            return ChangeDigest(session_id=session_id, ticker=ticker, changes=[])

        changes: List[ChangeItem] = []

        curr_dict = current_data.model_dump() if isinstance(current_data, SynthesizedOutput) else current_data
        prev_dict = previous_data.model_dump() if isinstance(previous_data, SynthesizedOutput) else previous_data

        curr_payload = curr_dict.get("raw_result_payload", {})
        prev_payload = prev_dict.get("raw_result_payload", {})

        # 1. Thesis Break Changes
        curr_breaks: List[Dict[str, Any]] = curr_payload.get("thesis_break_events", [])
        prev_breaks: List[Dict[str, Any]] = prev_payload.get("thesis_break_events", [])
        
        curr_assumptions = {b.get("broken_assumption") for b in curr_breaks if isinstance(b, dict)}
        prev_assumptions = {b.get("broken_assumption") for b in prev_breaks if isinstance(b, dict)}

        new_breaks = curr_assumptions - prev_assumptions
        for b in curr_breaks:
            if isinstance(b, dict) and b.get("broken_assumption") in new_breaks:
                changes.append(ChangeItem(
                    category="THESIS",
                    description=f"New Thesis Break: {b.get('broken_assumption')} - {b.get('explanation')}"
                ))

        # 2. Behavioral Drift Changes
        curr_drift = curr_payload.get("behavioral_drift_report", {})
        prev_drift = prev_payload.get("behavioral_drift_report", {})
        curr_sev = curr_drift.get("drift_severity", "NONE")
        prev_sev = prev_drift.get("drift_severity", "NONE")
        if curr_sev != prev_sev:
            changes.append(ChangeItem(
                category="RISK",
                description=f"Behavioral Drift severity shifted from {prev_sev} to {curr_sev}."
            ))

        # 3. Market Classification Changes
        curr_verdict = curr_dict.get("synthesized_verdict")
        prev_verdict = prev_dict.get("synthesized_verdict")
        if curr_verdict and prev_verdict and curr_verdict != prev_verdict:
            changes.append(ChangeItem(
                category="MARKET",
                description=f"Decision verdict updated from '{prev_verdict}' to '{curr_verdict}'."
            ))

        return ChangeDigest(
            session_id=session_id,
            ticker=ticker,
            changes=changes
        )

change_digest_engine = ChangeDigestEngine()
