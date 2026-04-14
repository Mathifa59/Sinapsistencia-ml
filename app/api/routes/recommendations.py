from fastapi import APIRouter, HTTPException

from app.domain.entities import (
    RecommendationRequest,
    RecommendationResponse,
    TrainingData,
    TrainingFromSupabaseRequest,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
)
from app.services.recommender import recommender_service

router = APIRouter()


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Genera recomendaciones de abogados para un médico.

    Acepta dos formatos de request:

    **Formato nativo (ML):**
    ```json
    {
      "doctor": {"id": "...", "name": "...", "specialty": "...", ...},
      "top_k": 10,
      "min_score": 0.0
    }
    ```

    **Formato frontend (Next.js):**
    ```json
    {
      "doctor_id": "uuid-del-doctor",
      "doctor_profile": {"specialty": "...", "hospital": "...", ...},
      "top_k": 10
    }
    ```

    El sistema selecciona automáticamente el modelo más adecuado:
    - **content**: médico sin historial de matches (cold-start)
    - **hybrid**: médico con historial previo (combina content + collaborative)

    La respuesta incluye explicabilidad (feature_importance + reasons) para cada
    recomendación, cumpliendo con los requisitos de transparencia del modelo.
    """
    try:
        return recommender_service.recommend(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/risk-assessment", response_model=RiskAssessmentResponse)
async def assess_risk(request: RiskAssessmentRequest) -> RiskAssessmentResponse:
    """
    Evalúa el riesgo médico-legal de un caso.

    Analiza múltiples factores (especialidad, complejidad, documentación,
    consentimiento, historial, temporalidad, prioridad) y produce:

    - **risk_score** (0-1): nivel global de riesgo
    - **risk_level**: bajo, moderado, alto, critico
    - **risk_factors**: desglose de cada factor con su peso y contribución
    - **recommendations**: acciones de mitigación recomendadas

    Épica 13 del Product Backlog (HU-044 a HU-047).
    """
    try:
        return recommender_service.assess_risk(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@router.post("/train-from-supabase")
async def train_from_supabase(request: TrainingFromSupabaseRequest) -> dict:
    """
    Entrena los modelos con datos reales desde Supabase.

    Lee perfiles de abogados de `lawyer_profiles` e interacciones de
    `contact_requests`, usando los UUIDs reales que coinciden con el frontend.

    Esto resuelve el problema de incompatibilidad entre los IDs de muestra
    (law_001, doc_001) y los UUIDs de producción.
    """
    try:
        return await recommender_service.train_from_supabase(
            supabase_url=request.supabase_url,
            supabase_key=request.supabase_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al entrenar desde Supabase: {exc}")


@router.get("/model/info")
async def get_model_info() -> dict:
    """Retorna el estado actual de los modelos entrenados."""
    if not recommender_service.is_ready:
        return {"status": "not_trained", "message": "Envía POST /train primero."}

    return {
        "status": "ready",
        "model_version": recommender_service.model_version,
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
        "risk_model": {
            "status": "ready",
            "version": recommender_service._risk_model._version,
        },
    }
