from fastapi import APIRouter, HTTPException

from app.domain.entities import RecommendationRequest, RecommendationResponse, TrainingData
from app.services.recommender import recommender_service

router = APIRouter()


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Genera recomendaciones de abogados para un médico.

    El sistema selecciona automáticamente el modelo más adecuado:
    - **content**: médico sin historial de matches (cold-start)
    - **hybrid**: médico con historial previo (combina content + collaborative)
    """
    try:
        return recommender_service.recommend(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/train")
async def train_model(training_data: TrainingData) -> dict:
    """
    Re-entrena el modelo con nuevos datos de abogados e interacciones.

    Enviar regularmente cuando se acumulen nuevos matches o se registren
    nuevos abogados en el sistema.
    """
    try:
        return recommender_service.train(training_data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/model/info")
async def get_model_info() -> dict:
    """Retorna el estado actual de los modelos entrenados."""
    if not recommender_service.is_ready:
        return {"status": "not_trained", "message": "Envía POST /train primero."}

    return {
        "status": "ready",
        "content_model": {
            "fitted": recommender_service._model.content_model.is_fitted,
            "num_lawyers": recommender_service._model.content_model.num_lawyers,
            "vocabulary_size": (
                recommender_service._model.content_model.vocabulary_size
                if recommender_service._model.content_model.is_fitted
                else 0
            ),
        },
        "collaborative_model": {
            "fitted": recommender_service._model.collaborative_model.is_fitted,
            "num_doctors": (
                recommender_service._model.collaborative_model.num_doctors
                if recommender_service._model.collaborative_model.is_fitted
                else 0
            ),
            "explained_variance": (
                recommender_service._model.collaborative_model.explained_variance_ratio
                if recommender_service._model.collaborative_model.is_fitted
                else 0.0
            ),
        },
    }
