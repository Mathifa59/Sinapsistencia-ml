"""
Entidades del dominio — modelos Pydantic que representan los objetos
que el sistema de recomendación y evaluación de riesgo reciben y producen.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ─── Perfiles ─────────────────────────────────────────────────────────────────


class DoctorProfile(BaseModel):
    """Perfil del médico usado como entrada para generar recomendaciones."""

    id: str
    name: str = Field(default="")
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


# ─── Request / Response de la API (Recomendaciones) ──────────────────────────


class RecommendationRequest(BaseModel):
    """
    Cuerpo del request POST /recommendations.
    Acepta el formato completo (doctor: DoctorProfile) O el formato
    simplificado que envía el frontend de Next.js (doctor_id + doctor_profile).
    """

    doctor: Optional[DoctorProfile] = Field(
        default=None,
        description="Perfil completo del doctor (formato nativo ML)",
    )
    # Formato alternativo enviado por el frontend Next.js
    doctor_id: Optional[str] = Field(default=None, description="UUID del doctor (formato frontend)")
    doctor_profile: Optional[dict] = Field(
        default=None,
        description="Perfil parcial del doctor (formato frontend: {specialty, sub_specialties, hospital, years_experience})",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Número de recomendaciones")
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score mínimo para incluir una recomendación",
    )

    def resolve_doctor(self) -> DoctorProfile:
        """
        Resuelve el perfil del doctor sin importar el formato recibido.
        Soporta tanto el formato nativo (doctor: DoctorProfile) como
        el formato del frontend (doctor_id + doctor_profile).
        """
        if self.doctor is not None:
            return self.doctor

        if self.doctor_id and self.doctor_profile:
            return DoctorProfile(
                id=self.doctor_id,
                name=self.doctor_profile.get("name", ""),
                specialty=self.doctor_profile.get("specialty", ""),
                sub_specialties=self.doctor_profile.get("sub_specialties", []),
                hospital=self.doctor_profile.get("hospital", ""),
                years_experience=self.doctor_profile.get("years_experience", 0),
                description=self.doctor_profile.get("description", ""),
            )

        raise ValueError(
            "Se requiere 'doctor' (formato nativo) o 'doctor_id' + 'doctor_profile' (formato frontend)."
        )


class FeatureImportance(BaseModel):
    """Importancia de una feature en la recomendación (explicabilidad)."""

    feature: str = Field(description="Nombre de la feature")
    importance: float = Field(description="Importancia relativa (0-1)")
    description: str = Field(default="", description="Descripción legible de la feature")


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
    feature_importance: list[FeatureImportance] = Field(
        default_factory=list,
        description="Top features que explican esta recomendación (explicabilidad)",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Razones legibles de la recomendación",
    )


class RecommendationResponse(BaseModel):
    """Respuesta completa del endpoint de recomendaciones."""

    doctor_id: str
    doctor_specialty: str
    recommendations: list[RecommendationScore]
    model_info: dict = Field(description="Metadatos del modelo usado en esta predicción")


# ─── Evaluación de Riesgo Médico-Legal (HU-044 a HU-047) ────────────────────


class RiskFactor(BaseModel):
    """Un factor individual que contribuye al riesgo del caso."""

    name: str = Field(description="Nombre del factor de riesgo")
    weight: float = Field(ge=0.0, le=1.0, description="Peso del factor en el riesgo total")
    value: float = Field(description="Valor del factor para este caso (0-1)")
    contribution: float = Field(description="Contribución al riesgo: weight × value")
    description: str = Field(default="", description="Explicación del factor")


class RiskAssessmentRequest(BaseModel):
    """Request para evaluar el riesgo médico-legal de un caso."""

    case_id: Optional[str] = Field(default=None, description="ID del caso (para tracking)")
    specialty: str = Field(description="Especialidad médica del caso")
    description: str = Field(default="", description="Descripción del caso")
    priority: str = Field(default="media", description="Prioridad: baja, media, alta, critica")
    patient_age: Optional[int] = Field(default=None, ge=0, le=120)
    patient_gender: Optional[str] = Field(default=None)
    has_prior_complaints: bool = Field(default=False, description="Historial de quejas previas")
    procedure_complexity: str = Field(
        default="media",
        description="Complejidad del procedimiento: baja, media, alta",
    )
    documentation_complete: bool = Field(
        default=True, description="Documentación clínica completa"
    )
    informed_consent: bool = Field(default=True, description="Consentimiento informado firmado")
    time_since_incident_days: Optional[int] = Field(
        default=None, ge=0, description="Días desde el incidente"
    )


class RiskAssessmentResponse(BaseModel):
    """Resultado de la evaluación de riesgo médico-legal."""

    case_id: Optional[str]
    risk_score: float = Field(ge=0.0, le=1.0, description="Score de riesgo global (0-1)")
    risk_level: str = Field(description="Nivel: bajo, moderado, alto, critico")
    risk_factors: list[RiskFactor] = Field(description="Factores individuales del riesgo")
    recommendations: list[str] = Field(
        description="Acciones recomendadas para mitigar el riesgo"
    )
    specialty_risk_baseline: float = Field(
        description="Riesgo base de la especialidad médica"
    )
    model_version: str = Field(description="Versión del modelo de riesgo")


# ─── Entrenamiento ────────────────────────────────────────────────────────────


class TrainingData(BaseModel):
    """Datos necesarios para entrenar/actualizar los modelos."""

    lawyers: list[LawyerProfile]
    interactions: list[Interaction] = Field(default_factory=list)


class TrainingFromSupabaseRequest(BaseModel):
    """Request para entrenar desde datos de Supabase."""

    supabase_url: str = Field(description="URL del proyecto Supabase")
    supabase_key: str = Field(description="Service role key de Supabase")
    retrain: bool = Field(default=True, description="Forzar re-entrenamiento")


class ModelMetrics(BaseModel):
    """Métricas de evaluación del sistema de recomendación."""

    precision_at_k: float = Field(description="Precision@K")
    recall_at_k: float = Field(description="Recall@K")
    ndcg_at_k: float = Field(description="Normalized Discounted Cumulative Gain@K")
    k: int
    num_test_doctors: int
    model: str
