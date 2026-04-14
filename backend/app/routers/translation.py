"""Translation routes."""
from fastapi import APIRouter, Depends
from typing import Dict

from app.schemas import TranslationRequest, TranslationResponse
from app.services.translation_service import TranslationService
from app.dependencies import get_current_user
from app.models import User

router = APIRouter(prefix="/translation", tags=["Translation"])

translation_service = TranslationService()


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    current_user: User = Depends(get_current_user)
):
    """Translate text to target language."""
    result = translation_service.translate(request.text, request.target_language)
    return TranslationResponse(**result)


@router.get("/languages")
async def get_languages() -> Dict[str, str]:
    """Get supported languages."""
    return translation_service.get_supported_languages()
