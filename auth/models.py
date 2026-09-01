from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator

class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must contain at least 8 characters")

    @field_validator("password")
    def validate_password_strength(cls, v: str) -> str:
        if len(v.strip()) < 8:
            raise ValueError("Password must contain at least 8 characters")
        return v

class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: str
    is_active: bool
    display_name: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str
    email: Optional[str] = None
    risk_profile: str = "conservative"
    portfolio_concentration: float = 0.12
    preferences: Dict[str, Any] = Field(default_factory=dict)

class UserProfileUpdateRequest(BaseModel):
    risk_profile: Optional[str] = None
    portfolio_concentration: Optional[float] = None
    preferences: Optional[Dict[str, Any]] = None

    @field_validator("risk_profile")
    def validate_risk_profile(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in ["conservative", "aggressive"]:
            raise ValueError("risk_profile must be either 'conservative' or 'aggressive'")
        return v.lower() if v is not None else None
