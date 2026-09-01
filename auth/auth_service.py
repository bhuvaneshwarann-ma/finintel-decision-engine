import os
import uuid
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES, APP_ENV

# Fallback development secret (cryptographically random per process lifecycle if not provided in .env)
_DEV_SECRET = secrets.token_urlsafe(32)

def get_jwt_secret() -> str:
    secret = JWT_SECRET_KEY.strip() if JWT_SECRET_KEY else ""
    if not secret:
        if APP_ENV.lower() == "production":
            raise RuntimeError("JWT_SECRET_KEY environment variable is missing in production environment!")
        return _DEV_SECRET
    return secret

class AuthService:
    def __init__(self):
        self._hasher = PasswordHasher(
            time_cost=2,
            memory_cost=65536,
            parallelism=2,
            hash_len=32,
            salt_len=16
        )
        # In-memory brute-force rate limiter: email -> list of failed timestamp floats (§24)
        self._failed_attempts: Dict[str, list[float]] = {}
        self._max_failed_attempts = 5
        self._lockout_window_seconds = 300  # 5 minutes

    # --- PASSWORD HASHING (§4) ---
    def hash_password(self, password: str) -> str:
        """Hashes plaintext password with Argon2. Never stores or logs plaintext."""
        return self._hasher.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verifies plaintext password against Argon2 hash in constant time."""
        try:
            return self._hasher.verify(password_hash, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    # --- JWT TOKEN MANAGEMENT (§5) ---
    def create_access_token(self, user_id: str, email: str, expires_delta: Optional[timedelta] = None) -> Tuple[str, int]:
        """Creates a signed JWT token containing user identity and expiration claims."""
        secret = get_jwt_secret()
        expire_minutes = JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        if expires_delta:
            expire_at = datetime.now(timezone.utc) + expires_delta
            expires_in_seconds = int(expires_delta.total_seconds())
        else:
            expire_at = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
            expires_in_seconds = expire_minutes * 60

        payload = {
            "sub": user_id,
            "email": email,
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int(expire_at.timestamp()),
            "jti": uuid.uuid4().hex
        }

        token = jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
        return token, expires_in_seconds

    def decode_access_token(self, token: str) -> Dict[str, Any]:
        """Decodes and validates a JWT token signature and expiration claims."""
        secret = get_jwt_secret()
        try:
            payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token signature or payload")

    # --- RATE LIMITING / BRUTE FORCE DEFENSE (§24) ---
    def check_rate_limit(self, email: str) -> bool:
        """
        Checks whether an email has exceeded the failed login threshold.
        Returns True if within limits, False if rate-limited.
        """
        now = time.time()
        key = email.lower().strip()
        attempts = self._failed_attempts.get(key, [])
        # Prune old attempts
        recent = [t for t in attempts if now - t < self._lockout_window_seconds]
        self._failed_attempts[key] = recent
        return len(recent) < self._max_failed_attempts

    def record_failed_login(self, email: str):
        """Records a failed login attempt for the specified email."""
        now = time.time()
        key = email.lower().strip()
        if key not in self._failed_attempts:
            self._failed_attempts[key] = []
        self._failed_attempts[key].append(now)

    def reset_failed_login(self, email: str):
        """Clears failed login history upon successful login."""
        key = email.lower().strip()
        if key in self._failed_attempts:
            del self._failed_attempts[key]

auth_service = AuthService()
