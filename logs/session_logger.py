import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

class SessionLogger:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._ticker_history: Dict[str, List[str]] = {}  # ticker -> list of session_ids

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
        raw_result_payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        entry = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ticker": ticker,
            "persona": persona,
            "agent_latencies_ms": agent_latencies,
            "total_latency_ms": round(total_latency_ms, 2),
            "agent_statuses": agent_statuses,
            "confidence": round(confidence, 3),
            "signal_classification": signal_classification,
            "synthesized_verdict": synthesized_verdict,
            "degraded_data": degraded_data,
            "source_count": source_count,
            "risk_concentration_score": round(risk_concentration_score, 3),
            "filing_freshness_flag_count": filing_freshness_flag_count,
            "devils_advocate_verdict_changed": devils_advocate_verdict_changed,
            "behavioral_flags_triggered_count": behavioral_flags_triggered_count,
            "raw_result_payload": raw_result_payload or {}
        }
        self._sessions[session_id] = entry
        if ticker not in self._ticker_history:
            self._ticker_history[ticker] = []
        self._ticker_history[ticker].append(session_id)
        return entry

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def get_previous_session_for_ticker(self, ticker: str, current_session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        history = self._ticker_history.get(ticker, [])
        if not history:
            return None
        filtered = [sid for sid in history if sid != current_session_id]
        if not filtered:
            return None
        return self._sessions.get(filtered[-1])

session_logger = SessionLogger()
