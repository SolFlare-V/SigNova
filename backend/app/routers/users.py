"""User management routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import UserResponse, UserUpdate
from app.services.user_service import UserService
from app.dependencies import get_current_user
from app.models import User

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/profile", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get user profile."""
    return current_user


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    return UserService.update_user(db, current_user.username, update_data)


@router.put("/language/{language}", response_model=UserResponse)
async def update_language(
    language: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update preferred language."""
    return UserService.update_language(db, current_user.username, language)


@router.put("/theme/{theme}", response_model=UserResponse)
async def update_theme(
    theme: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update theme preference."""
    return UserService.update_theme(db, current_user.username, theme)
