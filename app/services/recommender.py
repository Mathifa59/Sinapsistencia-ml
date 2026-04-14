"""
Servicio orquestador — gestiona el ciclo de vida de los modelos ML.
Carga, actualiza y expone el HybridRecommender y RiskAssessmentModel
al resto de la aplicación.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.models.hybrid import HybridRecommender
from app.models.risk_assessment import RiskAssessmentModel, risk_model
from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    Interaction,
    RecommendationRequest,
    RecommendationResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    TrainingData,
)
from app.config import settings, get_model_version_tag
from app.services.supabase_loader import (
    load_lawyers_from_supabase,
    load_interactions_from_supabase,
)

logger = logging.getLogger(__name__)


class RecommenderService:
    """
    Singleton que gestiona los modelos de recomendación y riesgo.

    Responsabilidades:
        - Inicializar los modelos al arrancar el servidor
        - Entrenar/re-entrenar con nuevos datos (muestra o Supabase)
        - Delegar predicciones al HybridRecommender y RiskAssessmentModel
        - Versionado de modelos entrenados
    """

    def __init__(self):
        self._model = HybridRecommender()
        self._risk_model = risk_model
        self._artifacts_path = settings.models_dir
        self._model_version: str | None = None
        self._trained_at: str | None = None

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
                # Cargar metadata de versión si existe
                meta_path = os.path.join(self._artifacts_path, "model_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    self._model_version = meta.get("version")
                    self._trained_at = meta.get("trained_at")
                logger.info("Modelos cargados exitosamente (version=%s).", self._model_version)
                return
            except Exception as exc:
                logger.warning("Error al cargar modelos: %s. Re-entrenando...", exc)

        # Si no hay modelos guardados, entrenar con datos de muestra
        logger.info("Entrenando modelos con datos de muestra...")
        sample_data = _load_sample_data()
        if sample_data:
            self._model.fit(
                lawyers=sample_data["lawyers"],
                interactions=sample_data["interactions"],
            )
            self._save_with_metadata()
            logger.info("Modelos entrenados y guardados.")
        else:
            logger.warning(
                "No se encontraron datos de muestra. El modelo no está entrenado."
            )

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Genera recomendaciones para un médico."""
        if not self._model.is_fitted:
            raise RuntimeError(
                "El modelo no está entrenado. "
                "Envía un POST /api/v1/train con datos de abogados primero."
            )

        # Resolver el perfil del doctor (soporta ambos formatos)
        doctor = request.resolve_doctor()

        recommendations = self._model.recommend(
            doctor=doctor,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        model_info = self._model.get_model_info(doctor.id)
        model_info["model_version"] = self._model_version
        model_info["trained_at"] = self._trained_at

        return RecommendationResponse(
            doctor_id=doctor.id,
            doctor_specialty=doctor.specialty,
            recommendations=recommendations,
            model_info=model_info,
        )

    def assess_risk(self, request: RiskAssessmentRequest) -> RiskAssessmentResponse:
        """Evalúa el riesgo médico-legal de un caso."""
        return self._risk_model.assess(request)

    def train(self, training_data: TrainingData) -> dict:
        """Re-entrena el modelo con nuevos datos y persiste los artefactos."""
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
        self._save_with_metadata()

        return {
            "status": "ok",
            "model_version": self._model_version,
            "trained_at": self._trained_at,
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

    async def train_from_supabase(
        self,
        supabase_url: str,
        supabase_key: str,
    ) -> dict:
        """
        Entrena los modelos con datos reales desde Supabase.
        Resuelve el problema de IDs: usa user_id (UUID) como ID del abogado,
        que es el mismo que envía el frontend.
        """
        logger.info("Cargando datos desde Supabase para entrenamiento...")

        lawyers = await load_lawyers_from_supabase(supabase_url, supabase_key)
        if not lawyers:
            raise ValueError("No se encontraron abogados en Supabase.")

        interactions = await load_interactions_from_supabase(supabase_url, supabase_key)

        logger.info(
            "Datos cargados: %d abogados, %d interacciones. Entrenando...",
            len(lawyers),
            len(interactions),
        )

        self._model = HybridRecommender()
        self._model.fit(lawyers=lawyers, interactions=interactions)
        self._save_with_metadata(source="supabase")

        return {
            "status": "ok",
            "source": "supabase",
            "model_version": self._model_version,
            "trained_at": self._trained_at,
            "lawyers_indexed": len(lawyers),
            "interactions_used": len(interactions),
            "content_model": "trained",
            "collaborative_model": (
                "trained" if self._model.collaborative_model.is_fitted
                else f"skipped ({len(interactions)} interacciones < 5 mínimo)"
            ),
        }

    def _save_with_metadata(self, source: str = "manual") -> None:
        """Guarda modelos + metadata de versión."""
        self._model_version = get_model_version_tag()
        self._trained_at = datetime.now(timezone.utc).isoformat()

        self._model.save(self._artifacts_path)

        # Guardar metadata
        meta = {
            "version": self._model_version,
            "trained_at": self._trained_at,
            "source": source,
            "app_version": settings.app_version,
        }
        meta_path = os.path.join(self._artifacts_path, "model_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @property
    def is_ready(self) -> bool:
        return self._model.is_fitted

    @property
    def model_version(self) -> str | None:
        return self._model_version


# Singleton — una sola instancia compartida por toda la app
recommender_service = RecommenderService()


# ─── Helpers de carga de datos de muestra ────────────────────────────────────


def _load_sample_data() -> dict | None:
    """Carga datos de muestra desde los archivos JSON del proyecto."""
    lawyers_path = Path("data/sample/lawyers.json")
    interactions_path = Path("data/sample/interactions.json")

    if not lawyers_path.exists():
        return None

    try:
        with open(lawyers_path, encoding="utf-8") as f:
            lawyers_raw = json.load(f)
        lawyers = [LawyerProfile(**l) for l in lawyers_raw]

        interactions: list[Interaction] = []
        if interactions_path.exists():
            with open(interactions_path, encoding="utf-8") as f:
                interactions_raw = json.load(f)
            interactions = [Interaction(**i) for i in interactions_raw]

        return {"lawyers": lawyers, "interactions": interactions}
    except Exception as exc:
        logger.error("Error cargando datos de muestra: %s", exc)
        return None
