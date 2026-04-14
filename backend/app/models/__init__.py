"""SQLAlchemy ORM models for the application."""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    """User model for authentication and profile management."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user")  # user, admin
    preferred_language = Column(String(10), default="en")
    theme = Column(String(20), default="dark")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class SessionLog(Base):
    """Session log model for tracking recognition sessions."""
    __tablename__ = "session_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    gesture_detected = Column(String(100))
    confidence = Column(Float)
    translated_text = Column(Text)
    target_language = Column(String(10))
    duration_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
