"""Application configuration using Pydantic Settings."""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "SignLang AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # JWT Authentication
    SECRET_KEY: str = "super-secret-key-change-in-production-abc123xyz"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Database
    DATABASE_URL: str = "sqlite:///./signlang.db"
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    # ML Model
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models", "gesture_model.pkl")
    LABELS_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models", "labels.pkl")
    
    # Translation
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: str = "en,ta,hi"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
