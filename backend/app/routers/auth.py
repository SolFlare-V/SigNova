"""Authentication routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import UserRegister, UserLogin, GoogleAuthRequest, Token, UserResponse
from app.services.auth_service import AuthService
from app.dependencies import get_current_user
from app.models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user account."""
    user = AuthService.register_user(db, user_data)
    return user


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login and receive JWT token."""
    return AuthService.login(db, user_data.username, user_data.password)


@router.post("/google", response_model=Token)
async def google_login(data: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Login or register via Google OAuth."""
    return AuthService.google_login(db, data.token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user profile."""
    return current_user

