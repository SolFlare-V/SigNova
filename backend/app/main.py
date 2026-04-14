"""
SignLang AI - FastAPI Application Entry Point

A Multilingual AI-based Real-Time Sign Language Recognition System.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.database import init_db
from app.middleware import error_handler_middleware, logging_middleware
from app.routers import auth_router, users_router, gesture_router, translation_router
from app.routers.collect import router as collect_router
from app.services.model_loader import ModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("signlang")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize database
    init_db()
    logger.info("✅ Database initialized")
    
    # Load ML model
    model_loader = ModelLoader()
    loaded = model_loader.load_model(settings.MODEL_PATH)
    if loaded:
        logger.info("✅ ML model loaded successfully")
    else:
        logger.warning("⚠️ ML model not found - running in demo mode")

    labels_loaded = model_loader.load_labels(settings.LABELS_PATH)
    if labels_loaded:
        logger.info("✅ ML labels loaded successfully")
    else:
        logger.warning("⚠️ Labels file not found - running in demo mode")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down SignLang AI")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Multilingual AI-based Real-Time Sign Language Recognition System",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.middleware("http")(logging_middleware)
app.middleware("http")(error_handler_middleware)

# Include routers
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(gesture_router, prefix="/api")
app.include_router(translation_router, prefix="/api")
app.include_router(collect_router, prefix="/api")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    model_loader = ModelLoader()
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded,
        "database": "connected"
    }
