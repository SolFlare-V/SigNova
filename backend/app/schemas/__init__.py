"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ─── Auth Schemas ─────────────────────────────────────────────
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class GoogleAuthRequest(BaseModel):
    token: str  # Google ID token from frontend

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


# ─── User Schemas ─────────────────────────────────────────────
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    preferred_language: str
    theme: str
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    preferred_language: Optional[str] = None
    theme: Optional[str] = None


# ─── Gesture Schemas ──────────────────────────────────────────
class GestureRequest(BaseModel):
    image_data: str  # Base64 encoded image

class GesturePrediction(BaseModel):
    gesture: str
    confidence: float
    landmarks_detected: bool

class GestureResponse(BaseModel):
    prediction: GesturePrediction
    translated_text: Optional[str] = None
    target_language: Optional[str] = None


# ─── Translation Schemas ──────────────────────────────────────
class TranslationRequest(BaseModel):
    text: str
    target_language: str = "ta"

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str


# ─── Session Log Schemas ──────────────────────────────────────
class SessionLogResponse(BaseModel):
    id: int
    gesture_detected: Optional[str]
    confidence: Optional[float]
    translated_text: Optional[str]
    target_language: Optional[str]
    duration_ms: Optional[int]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


# ─── Dashboard Schemas ────────────────────────────────────────
class DashboardStats(BaseModel):
    total_sessions: int
    total_gestures_detected: int
    average_confidence: float
    recent_activity: List[SessionLogResponse]
    system_status: str = "online"

class SystemStatus(BaseModel):
    model_loaded: bool
    model_name: str
    supported_gestures: List[str]
    supported_languages: List[str]
    status: str
