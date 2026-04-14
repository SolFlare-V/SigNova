"""Dependencies for FastAPI dependency injection."""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.models import User

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token."""
    token_data = AuthService.decode_token(credentials.credentials)
    user = UserService.get_user_by_username(db, token_data.username)
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user
