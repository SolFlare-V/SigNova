"""Request logging middleware."""
import time
import logging
from fastapi import Request

logger = logging.getLogger("signlang.api")


async def logging_middleware(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = time.time()
    
    logger.info(f"→ {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = round((time.time() - start_time) * 1000, 2)
    logger.info(f"← {request.method} {request.url.path} [{response.status_code}] {duration}ms")
    
    return response
