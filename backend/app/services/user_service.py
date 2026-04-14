"""User service for profile management."""
from sqlalchemy.orm import Session
from fastapi import HTTPException
from typing import Optional

from app.models import User
from app.schemas import UserUpdate


class UserService:
    """Handles user profile operations."""
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> User:
        """Get user by username."""
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> User:
        """Get user by ID."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    
    @staticmethod
    def update_user(db: Session, username: str, update_data: UserUpdate) -> User:
        """Update user profile."""
        user = UserService.get_user_by_username(db, username)
        
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(user, field, value)
        
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def update_language(db: Session, username: str, language: str) -> User:
        """Update user's preferred language."""
        user = UserService.get_user_by_username(db, username)
        user.preferred_language = language
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def update_theme(db: Session, username: str, theme: str) -> User:
        """Update user's theme preference."""
        user = UserService.get_user_by_username(db, username)
        user.theme = theme
        db.commit()
        db.refresh(user)
        return user
