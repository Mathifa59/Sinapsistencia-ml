"""
Servicio orquestador — gestiona el ciclo de vida de los modelos ML.
Carga, actualiza y expone el HybridRecommender al resto de la aplicación.
"""

import os
import json
import logging
from pathlib import Path

from app.models.hybrid import HybridRecommender
from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    Interaction,
    RecommendationRequest,
    RecommendationResponse,
    TrainingData,
)
from app.config import settings

logger = logging.getLogger(__name__)


class RecommenderService:
    """
    Singleton que gestiona el modelo híbrido.

    Responsabilidades:
        - Inicializar el modelo al arrancar el servidor
        - Entrenar/re-entrenar con nuevos datos
        - Delegar las predicciones al HybridRecommender
    """

    def __init__(self):
        self._model = HybridRecommender()
        self._artifacts_path = settings.models_dir

    async def initialize(self) -> None:
        """
        Inicializa el servicio al arrancar FastAPI.
        Intenta cargar modelos pre-entrenados; si no existen, entrena con datos de muestra.
        """
        artifacts_exist = os.path.exists(
            os.path.join(self._artifacts_path, "tfidf.pkl")
        )

        if artifacts_exist:
            logger.info("Cargando modelos pre-entrenados desde '%s'...", self._artifacts_path)
            try:
                self._model.load(self._artifacts_path)
                logger.info("Modelos cargados exitosamente.")
                return
            except Exception as exc:
                logger.warning("Error al cargar modelos: %s. Re-entrenando...", exc)

        # Si no hay modelos guardados, entrena con los datos de muestra
        logger.info("Entrenando modelos con datos de muestra...")
        sample_data = _load_sample_data()
        if sample_data:
            self._model.fit(
                lawyers=sample_data["lawyers"],
                interactions=sample_data["interactions"],
            )
            self._model.save(self._artifacts_path)
            logger.info("Modelos entrenados y guardados.")
        else:
            logger.warning(
                "No se encontraron datos de muestra. El modelo no está entrenado."
            )

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Genera recomendaciones para un médico.
        """
        if not self._model.is_fitted:
            raise RuntimeError(
                "El modelo no está entrenado. "
                "Envía un POST /api/v1/train con datos de abogados primero."
            )

        recommendations = self._model.recommend(
            doctor=request.doctor,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        model_info = self._model.get_model_info(request.doctor.id)

        return RecommendationResponse(
            doctor_id=request.doctor.id,
            doctor_specialty=request.doctor.specialty,
            recommendations=recommendations,
            model_info=model_info,
        )

    def train(self, training_data: TrainingData) -> dict:
        """
        Re-entrena el modelo con nuevos datos y persiste los artefactos.
        """
        logger.info(
            "Re-entrenando con %d abogados y %d interacciones...",
            len(training_data.lawyers),
            len(training_data.interactions),
        )
        self._model = HybridRecommender()
        self._model.fit(
            lawyers=training_data.lawyers,
            interactions=training_data.interactions,
        )
        self._model.save(self._artifacts_path)

        return {
            "status": "ok",
            "lawyers_indexed": len(training_data.lawyers),
            "interactions_used": len(training_data.interactions),
            "content_model": "trained",
            "collaborative_model": (
                "trained" if self._model.collaborative_model.is_fitted else "skipped (< 5 interacciones)"
            ),
            "vocabulary_size": (
                self._model.content_model.vocabulary_size
                if self._model.content_model.is_fitted
                else 0
            ),
        }

    @property
    def is_ready(self) -> bool:
        return self._model.is_fitted


# Singleton — una sola instancia compartida por toda la app
recommender_service = RecommenderService()


# ─── Helpers de carga de datos de muestra ────────────────────────────────────


def _load_sample_data() -> dict | None:
    """Carga datos de muestra desde los archivos JSON/CSV del proyecto."""
    lawyers_path = Path("data/sample/lawyers.json")
    interactions_path = Path("data/sample/interactions.json")

    if not lawyers_path.exists():
        return None

    try:
        with open(lawyers_path) as f:
            lawyers_raw = json.load(f)
        lawyers = [LawyerProfile(**l) for l in lawyers_raw]

        interactions: list[Interaction] = []
        if interactions_path.exists():
            with open(interactions_path) as f:
                interactions_raw = json.load(f)
            interactions = [Interaction(**i) for i in interactions_raw]

        return {"lawyers": lawyers, "interactions": interactions}
    except Exception as exc:
        logger.error("Error cargando datos de muestra: %s", exc)
        return None
