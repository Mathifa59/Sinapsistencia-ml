"""
Entidades del dominio — modelos Pydantic que representan los objetos
que el sistema de recomendación recibe y produce.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ─── Perfiles ─────────────────────────────────────────────────────────────────


class DoctorProfile(BaseModel):
    """Perfil del médico usado como entrada para generar recomendaciones."""

    id: str
    name: str
    specialty: str = Field(
        description="Especialidad médica principal (ej: 'Cardiología', 'Neurología')"
    )
    sub_specialties: list[str] = Field(
        default_factory=list,
        description="Sub-especialidades adicionales",
    )
    hospital: str = Field(default="", description="Hospital o clínica de trabajo")
    years_experience: int = Field(default=0, ge=0)
    description: str = Field(
        default="",
        description="Texto libre del perfil — usado para enriquecer el vector TF-IDF",
    )


class LawyerProfile(BaseModel):
    """Perfil del abogado indexado en el sistema de recomendación."""

    id: str
    name: str
    specialties: list[str] = Field(
        description="Áreas legales (ej: ['Responsabilidad Civil Médica', 'Derecho Penal Médico'])"
    )
    medical_areas: list[str] = Field(
        default_factory=list,
        description="Especialidades médicas que domina (ej: ['Cardiología', 'Neurología'])",
    )
    years_experience: int = Field(default=0, ge=0)
    resolved_cases: int = Field(default=0, ge=0, description="Casos resueltos exitosamente")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Valoración promedio (0-5)")
    description: str = Field(default="", description="Texto libre del perfil")


# ─── Interacciones (datos de entrenamiento del Collaborative Filtering) ───────


class Interaction(BaseModel):
    """
    Registro de una interacción doctor-abogado.
    Representa si un médico aceptó o rechazó una recomendación de abogado.
    Es el dato de entrenamiento del modelo de Collaborative Filtering.
    """

    doctor_id: str
    lawyer_id: str
    accepted: bool = Field(description="True = match exitoso, False = rechazado")
    rating: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Valoración explícita (1-5). Si None, se infiere del campo accepted.",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def implicit_score(self) -> float:
        """
        Convierte la interacción a un score numérico para la matriz de ratings.
        Si hay valoración explícita, la normaliza a [0, 1].
        Si no, usa 1.0 para aceptados y 0.0 para rechazados.
        """
        if self.rating is not None:
            return (self.rating - 1) / 4  # normaliza 1-5 → 0-1
        return 1.0 if self.accepted else 0.0


# ─── Request / Response de la API ─────────────────────────────────────────────


class RecommendationRequest(BaseModel):
    """Cuerpo del request POST /recommendations."""

    doctor: DoctorProfile
    top_k: int = Field(default=10, ge=1, le=50, description="Número de recomendaciones")
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score mínimo para incluir una recomendación",
    )


class RecommendationScore(BaseModel):
    """Una recomendación individual con su score de afinidad."""

    lawyer_id: str
    lawyer_name: str
    score: float = Field(ge=0.0, le=1.0, description="Score de afinidad (0-1)")
    content_score: float = Field(description="Score del modelo content-based")
    collaborative_score: float = Field(description="Score del modelo collaborative")
    matched_specialties: list[str] = Field(
        default_factory=list,
        description="Especialidades que coinciden entre doctor y abogado",
    )
    model_used: str = Field(
        description="Modelo que generó la recomendación: 'content', 'collaborative', 'hybrid'"
    )


class RecommendationResponse(BaseModel):
    """Respuesta completa del endpoint de recomendaciones."""

    doctor_id: str
    doctor_specialty: str
    recommendations: list[RecommendationScore]
    model_info: dict = Field(description="Metadatos del modelo usado en esta predicción")


# ─── Entrenamiento ────────────────────────────────────────────────────────────


class TrainingData(BaseModel):
    """Datos necesarios para entrenar/actualizar los modelos."""

    lawyers: list[LawyerProfile]
    interactions: list[Interaction] = Field(default_factory=list)


class ModelMetrics(BaseModel):
    """Métricas de evaluación del sistema de recomendación."""

    precision_at_k: float = Field(description="Precision@K")
    recall_at_k: float = Field(description="Recall@K")
    ndcg_at_k: float = Field(description="Normalized Discounted Cumulative Gain@K")
    k: int
    num_test_doctors: int
    model: str
