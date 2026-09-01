from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, status, Depends
from auth.auth_service import auth_service
from auth.database import auth_db

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    FastAPI dependency for authenticating protected routes via JWT (§8).
    Extracts Bearer token, verifies signature and expiration, and fetches active user.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = parts[1]
    try:
        payload = auth_service.decode_access_token(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing subject identifier.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = auth_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Safe return without exposing password_hash
    return {
        "id": user["id"],
        "email": user["email"],
        "created_at": user["created_at"],
        "is_active": bool(user["is_active"]),
        "display_name": user.get("display_name")
    }

async def get_optional_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Optional authentication dependency that returns user if valid token exists, None otherwise."""
    if not authorization:
        return None
    try:
        return await get_current_user(authorization=authorization)
    except HTTPException:
        return None
