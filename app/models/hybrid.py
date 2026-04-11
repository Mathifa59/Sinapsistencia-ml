"""
Modelo 3: Hybrid Recommender
============================
Combina Content-Based Filtering y Collaborative Filtering en un sistema unificado.

Estrategia de combinación — Weighted Hybrid:
    score_final(d, a) = α × score_content(d, a) + β × score_collaborative(d, a)

    donde α + β = 1

Ponderación adaptativa según disponibilidad de datos:
    ┌─────────────────────────────────────────┬──────────────────────────────┐
    │ Situación                               │ Pesos aplicados              │
    ├─────────────────────────────────────────┼──────────────────────────────┤
    │ Doctor sin historial (cold-start)       │ α=1.0, β=0.0 (solo content) │
    │ Doctor con poco historial (<10 matches) │ α=0.7, β=0.3               │
    │ Doctor con historial normal (10-50)     │ α=0.5, β=0.5               │
    │ Doctor con historial rico (>50 matches) │ α=0.3, β=0.7               │
    └─────────────────────────────────────────┴──────────────────────────────┘

¿Por qué híbrido y no solo uno de los dos?
    - Content-based solo: no captura preferencias implícitas, no mejora con el tiempo
    - Collaborative solo: falla totalmente con médicos nuevos (cold-start problem)
    - Híbrido: lo mejor de ambos mundos, transición suave de uno al otro

Esta combinación es el mismo principio que usan Netflix y Spotify para sus
sistemas de recomendación en producción.
"""

import numpy as np
from typing import Sequence

from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    Interaction,
    RecommendationScore,
)
from app.models.content_based import ContentBasedRecommender
from app.models.collaborative import CollaborativeRecommender
from app.data.preprocessing import find_matching_specialties
from app.config import settings


def _adaptive_weights(doctor_interaction_count: int) -> tuple[float, float]:
    """
    Calcula los pesos α (content) y β (collaborative) según el historial del doctor.

    A más historial, más confiamos en el colaborativo porque tiene más datos
    para aprender las preferencias reales del médico.

    Returns:
        (alpha, beta) donde alpha + beta = 1.0
    """
    if doctor_interaction_count == 0:
        return 1.0, 0.0      # cold-start: 100% content-based
    elif doctor_interaction_count < 10:
        return 0.7, 0.3      # poco historial: predomina content
    elif doctor_interaction_count < 50:
        return 0.5, 0.5      # historial normal: equilibrio
    else:
        return 0.3, 0.7      # historial rico: predomina collaborative


class HybridRecommender:
    """
    Sistema de recomendación híbrido que combina Content-Based y Collaborative.

    Atributos:
        content_model       : ContentBasedRecommender entrenado
        collaborative_model : CollaborativeRecommender entrenado (opcional)
        _interaction_counts : dict doctor_id → número de interacciones
        _lawyers            : Perfiles de abogados para lookup
    """

    def __init__(self):
        self.content_model = ContentBasedRecommender()
        self.collaborative_model = CollaborativeRecommender()
        self._interaction_counts: dict[str, int] = {}
        self._lawyers: dict[str, LawyerProfile] = {}

    # ─── Entrenamiento ────────────────────────────────────────────────────────

    def fit(
        self,
        lawyers: Sequence[LawyerProfile],
        interactions: Sequence[Interaction] | None = None,
    ) -> "HybridRecommender":
        """
        Entrena ambos modelos.

        El Content-Based siempre se entrena (solo necesita perfiles de abogados).
        El Collaborative solo se entrena si hay suficientes interacciones (≥5).

        Args:
            lawyers     : Perfiles de todos los abogados del sistema.
            interactions: Historial de matches. Puede ser None o vacío.

        Returns:
            self
        """
        if not lawyers:
            raise ValueError("Se necesita al menos un abogado para entrenar el modelo.")

        self._lawyers = {l.id: l for l in lawyers}

        # Siempre entrena content-based
        self.content_model.fit(lawyers)

        # Entrena collaborative solo si hay suficiente historial
        interactions = interactions or []
        if len(interactions) >= 5:
            self.collaborative_model.fit(interactions, lawyers)
            # Cuenta interacciones por doctor para ponderación adaptativa
            for interaction in interactions:
                self._interaction_counts[interaction.doctor_id] = (
                    self._interaction_counts.get(interaction.doctor_id, 0) + 1
                )

        return self

    # ─── Predicción ───────────────────────────────────────────────────────────

    def recommend(
        self,
        doctor: DoctorProfile,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[RecommendationScore]:
        """
        Genera las top-K recomendaciones usando el modelo híbrido.

        Pasos:
            1. Obtener scores content-based para todos los abogados
            2. Si el doctor tiene historial, obtener scores collaborative
            3. Combinar con pesos adaptativos (α, β)
            4. Ordenar por score combinado y retornar top-K

        Args:
            doctor   : Perfil del médico solicitante.
            top_k    : Número de recomendaciones a retornar.
            min_score: Filtro de score mínimo.

        Returns:
            Lista de RecommendationScore ordenada de mayor a menor.
        """
        interaction_count = self._interaction_counts.get(doctor.id, 0)
        alpha, beta = _adaptive_weights(interaction_count)

        # ── Scores content-based ──────────────────────────────────────────────
        # Pedimos más que top_k para tener pool completo al combinar
        content_results = self.content_model.recommend(
            doctor, top_k=len(self._lawyers), min_score=0.0
        )
        content_scores: dict[str, float] = {
            r.lawyer_id: r.content_score for r in content_results
        }

        # ── Scores collaborative (si aplica) ──────────────────────────────────
        collaborative_scores: dict[str, float] = {}
        if beta > 0 and self.collaborative_model.is_fitted:
            collab_results = self.collaborative_model.recommend(
                doctor.id, top_k=len(self._lawyers), min_score=0.0
            )
            collaborative_scores = {
                r.lawyer_id: r.collaborative_score for r in collab_results
            }

        # ── Combinar scores ───────────────────────────────────────────────────
        all_lawyer_ids = set(content_scores.keys()) | set(collaborative_scores.keys())

        combined: list[tuple[str, float, float, float]] = []
        for lawyer_id in all_lawyer_ids:
            c_score = content_scores.get(lawyer_id, 0.0)
            cf_score = collaborative_scores.get(lawyer_id, 0.0)
            final_score = alpha * c_score + beta * cf_score
            combined.append((lawyer_id, final_score, c_score, cf_score))

        # Ordena por score final
        combined.sort(key=lambda x: x[1], reverse=True)

        # ── Construir respuesta ───────────────────────────────────────────────
        model_label = (
            "content" if beta == 0.0
            else "collaborative" if alpha == 0.0
            else "hybrid"
        )

        results: list[RecommendationScore] = []
        for lawyer_id, final_score, c_score, cf_score in combined:
            if final_score < min_score:
                break
            if len(results) >= top_k:
                break

            lawyer = self._lawyers.get(lawyer_id)
            if not lawyer:
                continue

            results.append(
                RecommendationScore(
                    lawyer_id=lawyer_id,
                    lawyer_name=lawyer.name,
                    score=round(final_score, 4),
                    content_score=round(c_score, 4),
                    collaborative_score=round(cf_score, 4),
                    matched_specialties=find_matching_specialties(doctor, lawyer),
                    model_used=model_label,
                )
            )

        return results

    def get_model_info(self, doctor_id: str) -> dict:
        """
        Retorna metadatos sobre qué modelo se usará para este doctor
        y por qué. Útil para explicabilidad de las recomendaciones.
        """
        interaction_count = self._interaction_counts.get(doctor_id, 0)
        alpha, beta = _adaptive_weights(interaction_count)

        model_label = (
            "content_only" if beta == 0.0
            else "collaborative_only" if alpha == 0.0
            else "hybrid"
        )

        return {
            "model": model_label,
            "alpha_content": alpha,
            "beta_collaborative": beta,
            "doctor_interaction_count": interaction_count,
            "content_model_fitted": self.content_model.is_fitted,
            "collaborative_model_fitted": self.collaborative_model.is_fitted,
            "collaborative_model_has_doctor": (
                self.collaborative_model.has_doctor(doctor_id)
                if self.collaborative_model.is_fitted
                else False
            ),
            "vocabulary_size": (
                self.content_model.vocabulary_size
                if self.content_model.is_fitted
                else 0
            ),
            "svd_explained_variance": (
                self.collaborative_model.explained_variance_ratio
                if self.collaborative_model.is_fitted
                else 0.0
            ),
        }

    def save(self, path: str) -> None:
        """Serializa ambos modelos."""
        import joblib, os
        os.makedirs(path, exist_ok=True)
        self.content_model.save(path)
        if self.collaborative_model.is_fitted:
            self.collaborative_model.save(path)
        joblib.dump(self._interaction_counts, os.path.join(path, "interaction_counts.pkl"))
        joblib.dump(self._lawyers, os.path.join(path, "lawyers_hybrid.pkl"))

    def load(self, path: str) -> "HybridRecommender":
        """Carga ambos modelos desde disco."""
        import joblib, os
        self.content_model.load(path)
        collab_path = os.path.join(path, "svd.pkl")
        if os.path.exists(collab_path):
            self.collaborative_model.load(path)
        self._interaction_counts = joblib.load(os.path.join(path, "interaction_counts.pkl"))
        self._lawyers = joblib.load(os.path.join(path, "lawyers_hybrid.pkl"))
        return self

    @property
    def is_fitted(self) -> bool:
        return self.content_model.is_fitted
