import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import BASE_DIR

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
AUDIT_LOG_FILE = LOGS_DIR / "decision_audit_trail.jsonl"

class SessionLogger:
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def create_session_id(self) -> str:
        return f"sess_{uuid.uuid4().hex[:12]}"

    def log_session(
        self,
        session_id: str,
        ticker: str,
        persona: str,
        agent_latencies: Dict[str, float],
        total_latency_ms: float,
        agent_statuses: Dict[str, str],
        confidence: float,
        signal_classification: str,
        synthesized_verdict: str,
        degraded_data: bool,
        source_count: int,
        risk_concentration_score: float,
        filing_freshness_flag_count: int,
        devils_advocate_verdict_changed: bool,
        behavioral_flags_triggered_count: int,
        raw_result_payload: Dict[str, Any],
        llm_used: bool = False,
        fallback_used: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Logs session telemetry with optional user_id isolation (§12).
        Never logs passwords, tokens, or raw secrets.
        """
        log_entry = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "persona": persona,
            "total_latency_ms": round(total_latency_ms, 2),
            "agent_latencies_ms": agent_latencies,
            "agent_statuses": agent_statuses,
            "confidence": round(confidence, 3),
            "signal_classification": signal_classification,
            "synthesized_verdict": synthesized_verdict,
            "degraded_data": degraded_data,
            "llm_used": llm_used,
            "fallback_used": fallback_used,
            "source_count": source_count,
            "risk_concentration_score": round(risk_concentration_score, 3),
            "filing_freshness_flag_count": filing_freshness_flag_count,
            "devils_advocate_verdict_changed": devils_advocate_verdict_changed,
            "behavioral_flags_triggered_count": behavioral_flags_triggered_count
        }

        self.active_sessions[session_id] = {**log_entry, "payload": raw_result_payload}

        try:
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

        return log_entry

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.active_sessions.get(session_id)

session_logger = SessionLogger()
