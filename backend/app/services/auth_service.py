"""Authentication service with JWT token management."""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.config import get_settings
from app.models import User
from app.schemas import UserRegister, Token, TokenData

settings = get_settings()


class AuthService:
    """Handles user authentication, password hashing, and JWT tokens."""
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a plain text password."""
        return AuthService.pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return AuthService.pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> TokenData:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role", "user")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            return TokenData(username=username, role=role)
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    
    @staticmethod
    def register_user(db: Session, user_data: UserRegister) -> User:
        """Register a new user."""
        # Check existing username
        if db.query(User).filter(User.username == user_data.username).first():
            raise HTTPException(status_code=400, detail="Username already registered")
        
        # Check existing email
        if db.query(User).filter(User.email == user_data.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=AuthService.hash_password(user_data.password)
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> User:
        """Authenticate a user with username and password."""
        user = db.query(User).filter(User.username == username).first()
        if not user or not AuthService.verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Account is deactivated")
        return user
    
    @staticmethod
    def login(db: Session, username: str, password: str) -> Token:
        """Login and return access token."""
        user = AuthService.authenticate_user(db, username, password)
        token = AuthService.create_access_token(
            data={"sub": user.username, "role": user.role}
        )
        return Token(access_token=token)
    
    @staticmethod
    def google_login(db: Session, google_token: str) -> Token:
        """Verify Google ID token and login/register user."""
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        
        try:
            # Verify the Google ID token
            idinfo = id_token.verify_oauth2_token(
                google_token,
                google_requests.Request(),
                settings.GOOGLE_CLIENT_ID
            )
            
            # Extract user info from token
            email = idinfo.get('email')
            full_name = idinfo.get('name', email.split('@')[0])
            
            if not email:
                raise HTTPException(status_code=400, detail="Google account has no email")
            
            # Check if user exists by email
            user = db.query(User).filter(User.email == email).first()
            
            if not user:
                # Create new user from Google account
                # Generate a unique username from email
                base_username = email.split('@')[0]
                username = base_username
                counter = 1
                while db.query(User).filter(User.username == username).first():
                    username = f"{base_username}{counter}"
                    counter += 1
                
                user = User(
                    username=username,
                    email=email,
                    full_name=full_name,
                    hashed_password=AuthService.hash_password(f"google_oauth_{email}"),  # Placeholder password
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            
            if not user.is_active:
                raise HTTPException(status_code=400, detail="Account is deactivated")
            
            # Generate JWT token
            token = AuthService.create_access_token(
                data={"sub": user.username, "role": user.role}
            )
            return Token(access_token=token)
            
        except ValueError as e:
            raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")

