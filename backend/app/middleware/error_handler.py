"""Error handling middleware."""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback

logger = logging.getLogger(__name__)


async def error_handler_middleware(request: Request, call_next):
    """Global error handling middleware."""
    try:
        response = await call_next(request)
        return response
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail}
        )
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
