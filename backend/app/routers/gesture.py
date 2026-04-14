"""Gesture recognition routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import GestureRequest, GestureResponse, SystemStatus, SessionLogResponse, DashboardStats
from app.services.gesture_service import GestureService
from app.services.translation_service import TranslationService
from app.services.model_loader import ModelLoader
from app.dependencies import get_current_user
from app.models import User, SessionLog
from typing import List

router = APIRouter(prefix="/gesture", tags=["Gesture Recognition"])

gesture_service = GestureService()
translation_service = TranslationService()
model_loader = ModelLoader()


@router.post("/predict", response_model=GestureResponse)
async def predict_gesture(
    request: GestureRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict gesture from webcam frame."""
    import time
    start = time.time()
    
    labels = model_loader.labels or gesture_service.GESTURE_LABELS
    prediction = gesture_service.predict(request.image_data, model_loader.model, labels)
    
    # Translate if user has a non-English preference
    translated_text = None
    target_lang = current_user.preferred_language
    
    if target_lang != "en" and prediction.gesture not in ["nothing", "unknown"]:
        result = translation_service.translate(prediction.gesture, target_lang)
        translated_text = result["translated_text"]
    
    duration = int((time.time() - start) * 1000)
    
    # Log session
    log = SessionLog(
        user_id=current_user.id,
        gesture_detected=prediction.gesture,
        confidence=prediction.confidence,
        translated_text=translated_text,
        target_language=target_lang,
        duration_ms=duration
    )
    db.add(log)
    db.commit()
    
    return GestureResponse(
        prediction=prediction,
        translated_text=translated_text,
        target_language=target_lang
    )


@router.post("/predict/public", response_model=GestureResponse)
async def predict_gesture_public(request: GestureRequest):
    """
    Predict gesture from webcam frame — no authentication required.
    Used by the frontend real-time recognition loop.
    """
    labels = model_loader.labels or gesture_service.GESTURE_LABELS
    prediction = gesture_service.predict(request.image_data, model_loader.model, labels)
    return GestureResponse(prediction=prediction)


@router.post("/reset")
async def reset_smoothing_buffer():
    """Clear the temporal smoothing buffer (call when starting a new session)."""
    gesture_service.clear_buffer()
    return {"status": "buffer cleared"}


@router.post("/debug/detect")
async def debug_detect(request: GestureRequest):
    """
    Debug endpoint — returns raw MediaPipe detection info without prediction.
    Useful for diagnosing hand detection issues.
    """
    import base64, cv2, numpy as np

    # Decode image
    image_data = request.image_data
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if raw is None:
        return {"error": "Failed to decode image"}

    h, w = raw.shape[:2]

    # Run the same preprocessing as gesture_service
    img = raw.copy()
    if h < 224 or w < 224:
        scale = max(224 / h, 224 / w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run MediaPipe
    hands_detected = False
    num_landmarks = 0
    mp_api = "unavailable"
    if gesture_service.hand_landmarker is not None and getattr(gesture_service, '_mp_module', None):
        mp_api = "tasks"
        mp = gesture_service._mp_module
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = gesture_service.hand_landmarker.detect(mp_img)
        if results.hand_landmarks:
            hands_detected = True
            num_landmarks = len(results.hand_landmarks[0])

    return {
        "raw_size": {"h": h, "w": w},
        "processed_size": {"h": img.shape[0], "w": img.shape[1]},
        "mean_brightness_raw": round(float(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).mean()), 1),
        "mean_brightness_processed": round(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()), 1),
        "mediapipe_api": mp_api,
        "hand_detected": hands_detected,
        "num_landmarks": num_landmarks,
        "model_loaded": model_loader.is_loaded,
        "hands_initialized": gesture_service.hand_landmarker is not None,
    }



async def get_system_status():
    """Get AI system status."""
    return SystemStatus(
        model_loaded=model_loader.is_loaded,
        model_name=model_loader.model_name,
        supported_gestures=model_loader.get_supported_gestures(),
        supported_languages=list(translation_service.SUPPORTED_LANGUAGES.keys()),
        status="online" if model_loader.is_loaded else "demo_mode"
    )


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics for the current user."""
    logs = db.query(SessionLog).filter(
        SessionLog.user_id == current_user.id
    ).order_by(SessionLog.created_at.desc()).limit(10).all()
    
    total = db.query(SessionLog).filter(SessionLog.user_id == current_user.id).count()
    
    gestures = db.query(SessionLog).filter(
        SessionLog.user_id == current_user.id,
        SessionLog.gesture_detected.notin_(["nothing", "unknown"])
    ).count()
    
    from sqlalchemy import func
    avg_conf = db.query(func.avg(SessionLog.confidence)).filter(
        SessionLog.user_id == current_user.id
    ).scalar() or 0.0
    
    return DashboardStats(
        total_sessions=total,
        total_gestures_detected=gestures,
        average_confidence=round(float(avg_conf), 3),
        recent_activity=logs,
        system_status="online" if model_loader.is_loaded else "demo_mode"
    )


@router.get("/history", response_model=List[SessionLogResponse])
async def get_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recognition history."""
    logs = db.query(SessionLog).filter(
        SessionLog.user_id == current_user.id
    ).order_by(SessionLog.created_at.desc()).limit(limit).all()
    return logs
