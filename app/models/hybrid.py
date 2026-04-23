"""
Recomendador — Content-Based Filtering
=======================================
Recomienda abogados a médicos usando TF-IDF + similitud del coseno.

Este módulo expone HybridRecommender como fachada del ContentBasedRecommender
para mantener compatibilidad con el resto de la aplicación.

Algoritmo seleccionado: Content-Based (TF-IDF + coseno)
    - Justificación: herramienta correcta para matching semántico de perfiles textuales.
    - El Collaborative Filtering (SVD) fue descartado por requerir cientos de
      interacciones reales para funcionar; con el dataset actual introduce ruido.
    - Referencia benchmarking: sección de algoritmos ML, criterio de robustez.
"""

import joblib
import os
from typing import Sequence

from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    Interaction,
    RecommendationScore,
)
from app.models.content_based import ContentBasedRecommender
from app.models.explainability import explain_content_recommendation
from app.data.preprocessing import find_matching_specialties
from app.config import settings


class HybridRecommender:
    """
    Fachada del sistema de recomendación.

    Delega toda la lógica al ContentBasedRecommender (TF-IDF + coseno).
    Acepta interacciones en fit() por compatibilidad de interfaz, pero no las usa.
    """

    def __init__(self):
        self.content_model = ContentBasedRecommender()
        self._lawyers: dict[str, LawyerProfile] = {}

    # ─── Entrenamiento ────────────────────────────────────────────────────

    def fit(
        self,
        lawyers: Sequence[LawyerProfile],
        interactions: Sequence[Interaction] | None = None,
    ) -> "HybridRecommender":
        """
        Entrena el modelo content-based.

        Args:
            lawyers     : Perfiles de abogados a indexar.
            interactions: Ignorado (mantenido por compatibilidad).

        Returns:
            self
        """
        if not lawyers:
            raise ValueError("Se necesita al menos un abogado para entrenar el modelo.")

        self._lawyers = {l.id: l for l in lawyers}
        self.content_model.fit(lawyers)
        return self

    # ─── Predicción ───────────────────────────────────────────────────────

    def recommend(
        self,
        doctor: DoctorProfile,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[RecommendationScore]:
        """
        Genera las top-K recomendaciones de abogados para un médico.

        Args:
            doctor   : Perfil del médico solicitante.
            top_k    : Número de recomendaciones a retornar.
            min_score: Filtro de score mínimo.

        Returns:
            Lista de RecommendationScore ordenada de mayor a menor.
        """
        content_results = self.content_model.recommend(
            doctor, top_k=len(self._lawyers), min_score=0.0
        )

        # Normalizar scores al rango visible [0.55, 0.95]
        top_slice = content_results[:top_k]
        if len(top_slice) > 1:
            raw_scores = [r.score for r in top_slice]
            lo, hi = min(raw_scores), max(raw_scores)
            if hi > lo:
                for r in top_slice:
                    r.score = round(0.55 + ((r.score - lo) / (hi - lo)) * 0.40, 4)
                    r.content_score = r.score
            else:
                for r in top_slice:
                    r.score = 0.70
                    r.content_score = 0.70

        results: list[RecommendationScore] = []
        for r in top_slice:
            if r.score < min_score:
                break

            lawyer = self._lawyers.get(r.lawyer_id)
            if not lawyer:
                continue

            feature_importance, reasons = explain_content_recommendation(
                doctor=doctor,
                lawyer=lawyer,
                content_model=self.content_model,
                score=r.score,
            )

            results.append(
                RecommendationScore(
                    lawyer_id=r.lawyer_id,
                    lawyer_name=r.lawyer_name,
                    score=r.score,
                    content_score=r.score,
                    collaborative_score=0.0,
                    matched_specialties=find_matching_specialties(doctor, lawyer),
                    model_used="content",
                    feature_importance=feature_importance,
                    reasons=reasons,
                )
            )

        return results

    def get_model_info(self, doctor_id: str) -> dict:
        """Retorna metadatos del modelo para esta predicción."""
        return {
            "model": "content",
            "alpha_content": 1.0,
            "beta_collaborative": 0.0,
            "content_model_fitted": self.content_model.is_fitted,
            "vocabulary_size": (
                self.content_model.vocabulary_size
                if self.content_model.is_fitted
                else 0
            ),
        }

    # ─── Serialización ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serializa el modelo a disco."""
        os.makedirs(path, exist_ok=True)
        self.content_model.save(path)
        joblib.dump(self._lawyers, os.path.join(path, "lawyers_hybrid.pkl"))

    def load(self, path: str) -> "HybridRecommender":
        """Carga el modelo desde disco."""
        self.content_model.load(path)
        lawyers_path = os.path.join(path, "lawyers_hybrid.pkl")
        if os.path.exists(lawyers_path):
            self._lawyers = joblib.load(lawyers_path)
        return self

    @property
    def is_fitted(self) -> bool:
        return self.content_model.is_fitted
