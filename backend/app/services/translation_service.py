"""Translation service for multilingual support."""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TranslationService:
    """Handles text translation between supported languages."""
    
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "ta": "Tamil",
        "hi": "Hindi"
    }
    
    def __init__(self):
        """Initialize translation service."""
        self.translator = None
        self._init_translator()
    
    def _init_translator(self):
        """Initialize the Google Translator."""
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator
            logger.info("Translation service initialized (deep-translator)")
        except ImportError:
            logger.warning("deep-translator not available. Using fallback dictionary.")
            self.translator = None
    
    def translate(self, text: str, target_language: str = "ta") -> Dict[str, str]:
        """Translate text to the target language."""
        if target_language not in self.SUPPORTED_LANGUAGES:
            target_language = "ta"
        
        if target_language == "en":
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": "en",
                "target_language": "en"
            }
        
        translated = self._do_translate(text, target_language)
        
        return {
            "original_text": text,
            "translated_text": translated,
            "source_language": "en",
            "target_language": target_language
        }
    
    def _do_translate(self, text: str, target_language: str) -> str:
        """Perform the actual translation."""
        if self.translator is not None:
            try:
                result = self.translator(source='en', target=target_language).translate(text)
                return result
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return self._fallback_translate(text, target_language)
        
        return self._fallback_translate(text, target_language)
    
    def _fallback_translate(self, text: str, target_language: str) -> str:
        """Fallback dictionary-based translation for common sign language words."""
        fallback = {
            "ta": {
                "hello": "வணக்கம்", "thank you": "நன்றி", "yes": "ஆம்",
                "no": "இல்லை", "please": "தயவுசெய்து", "help": "உதவி",
                "water": "தண்ணீர்", "food": "உணவு", "good": "நல்ல",
                "bad": "கெட்ட", "sorry": "மன்னிக்கவும்", "love": "அன்பு",
                "friend": "நண்பர்", "family": "குடும்பம்", "name": "பெயர்",
            },
            "hi": {
                "hello": "नमस्ते", "thank you": "धन्यवाद", "yes": "हाँ",
                "no": "नहीं", "please": "कृपया", "help": "मदद",
                "water": "पानी", "food": "खाना", "good": "अच्छा",
                "bad": "बुरा", "sorry": "क्षमा करें", "love": "प्यार",
                "friend": "दोस्त", "family": "परिवार", "name": "नाम",
            }
        }
        
        lang_dict = fallback.get(target_language, {})
        lower_text = text.lower()
        
        if lower_text in lang_dict:
            return lang_dict[lower_text]
        
        return f"[{self.SUPPORTED_LANGUAGES.get(target_language, target_language)}] {text}"
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Return supported languages."""
        return self.SUPPORTED_LANGUAGES
