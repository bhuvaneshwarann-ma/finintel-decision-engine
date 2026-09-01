import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from config import AUTH_DB_PATH

class AuthDatabase:
    def __init__(self, db_path: str = AUTH_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    display_name TEXT
                )
            """)
            # 2. User Profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    risk_profile TEXT NOT NULL DEFAULT 'conservative',
                    portfolio_concentration REAL NOT NULL DEFAULT 0.12,
                    preferences_json TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            # 3. User Theses table (User-isolated investment thesis records §11)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_theses (
                    thesis_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    stated_reasons_json TEXT NOT NULL,
                    key_assumptions_json TEXT NOT NULL,
                    invalidating_conditions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Auto-seed demo investor account for instant one-click terminal login
            cursor.execute("SELECT id FROM users WHERE email = 'investor@example.com'")
            if not cursor.fetchone():
                try:
                    from auth.auth_service import auth_service
                    demo_hash = auth_service.hash_password("DemoPassword123!")
                    demo_id = "usr_demo_investor"
                    now_str = datetime.now(timezone.utc).isoformat()
                    cursor.execute(
                        "INSERT INTO users (id, email, password_hash, created_at, is_active, display_name) VALUES (?, 'investor@example.com', ?, ?, 1, 'Demo Investor')",
                        (demo_id, demo_hash, now_str)
                    )
                    cursor.execute(
                        "INSERT INTO user_profiles (user_id, risk_profile, portfolio_concentration, preferences_json) VALUES (?, 'conservative', 0.12, '{}')",
                        (demo_id,)
                    )
                except Exception:
                    pass

            conn.commit()

    # --- USER CRUD ---
    def create_user(self, user_id: str, email: str, password_hash: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (id, email, password_hash, created_at, is_active, display_name) VALUES (?, ?, ?, ?, 1, ?)",
                (user_id, email.lower().strip(), password_hash, now_str, display_name)
            )
            # Initialize default user profile
            cursor.execute(
                "INSERT INTO user_profiles (user_id, risk_profile, portfolio_concentration, preferences_json) VALUES (?, 'conservative', 0.12, '{}')",
                (user_id,)
            )
            conn.commit()
            return {
                "id": user_id,
                "email": email.lower().strip(),
                "created_at": now_str,
                "is_active": True,
                "display_name": display_name
            }

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    # --- PROFILE CRUD ---
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                r = dict(row)
                prefs = json.loads(r.get("preferences_json") or "{}")
                return {
                    "user_id": user_id,
                    "risk_profile": r["risk_profile"],
                    "portfolio_concentration": float(r["portfolio_concentration"]),
                    "preferences": prefs
                }
            # Fallback default
            return {
                "user_id": user_id,
                "risk_profile": "conservative",
                "portfolio_concentration": 0.12,
                "preferences": {}
            }

    def update_user_profile(
        self,
        user_id: str,
        risk_profile: Optional[str] = None,
        portfolio_concentration: Optional[float] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        curr = self.get_user_profile(user_id)
        new_risk = risk_profile or curr["risk_profile"]
        new_conc = portfolio_concentration if portfolio_concentration is not None else curr["portfolio_concentration"]
        new_prefs = preferences if preferences is not None else curr["preferences"]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_profiles (user_id, risk_profile, portfolio_concentration, preferences_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    risk_profile=excluded.risk_profile,
                    portfolio_concentration=excluded.portfolio_concentration,
                    preferences_json=excluded.preferences_json
                """,
                (user_id, new_risk, new_conc, json.dumps(new_prefs))
            )
            conn.commit()

        return {
            "user_id": user_id,
            "risk_profile": new_risk,
            "portfolio_concentration": new_conc,
            "preferences": new_prefs
        }

    # --- USER-ISOLATED THESIS CRUD ---
    def save_user_thesis(
        self,
        thesis_id: str,
        user_id: str,
        ticker: str,
        stated_reasons: List[str],
        key_assumptions: List[str],
        invalidating_conditions: List[str]
    ) -> Dict[str, Any]:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_theses (thesis_id, user_id, ticker, stated_reasons_json, key_assumptions_json, invalidating_conditions_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (thesis_id, user_id, ticker, json.dumps(stated_reasons), json.dumps(key_assumptions), json.dumps(invalidating_conditions), now_str)
            )
            conn.commit()

        return {
            "thesis_record_id": thesis_id,
            "ticker": ticker,
            "user_id": user_id,
            "created_at": now_str,
            "stated_reasons": stated_reasons,
            "key_assumptions": key_assumptions,
            "invalidating_conditions": invalidating_conditions
        }

    def get_user_theses_for_ticker(self, user_id: str, ticker: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_theses WHERE user_id = ? AND ticker = ? ORDER BY created_at DESC LIMIT 1",
                (user_id, ticker)
            )
            row = cursor.fetchone()
            if row:
                r = dict(row)
                return {
                    "thesis_record_id": r["thesis_id"],
                    "ticker": r["ticker"],
                    "user_id": r["user_id"],
                    "created_at": r["created_at"],
                    "stated_reasons": json.loads(r["stated_reasons_json"]),
                    "key_assumptions": json.loads(r["key_assumptions_json"]),
                    "invalidating_conditions": json.loads(r["invalidating_conditions_json"])
                }
            return None

auth_db = AuthDatabase()
