"""
Modelo 4: Evaluación de Riesgo Médico-Legal
=============================================
Evalúa el riesgo legal de un caso médico usando un modelo basado en reglas
ponderadas + Random Forest para casos con datos históricos.

Épica 13 del Product Backlog:
    HU-044: Evaluación automática de riesgo de un caso
    HU-045: Factores de riesgo desglosados con pesos
    HU-046: Recomendaciones de mitigación
    HU-047: Historial de evaluaciones de riesgo

Algoritmo:
    1. Se calcula un risk_score basado en factores ponderados:
        - Riesgo base por especialidad médica (algunas tienen más demandas)
        - Complejidad del procedimiento
        - Documentación clínica completa
        - Consentimiento informado
        - Historial de quejas previas
        - Tiempo desde el incidente
        - Prioridad asignada al caso

    2. Cada factor aporta un porcentaje al riesgo total.

    3. Se generan recomendaciones de mitigación basadas en los factores.

¿Por qué modelo basado en reglas + pesos?
    Con pocos datos históricos, un Random Forest o regresión logística
    sobreajustaría. Las reglas ponderadas son:
    - Interpretables (cada factor tiene un peso explícito)
    - Basadas en literatura médico-legal
    - Funcionales desde el primer caso (no requieren entrenamiento)
    - Extendibles cuando haya datos suficientes para un modelo supervisado
"""

import logging
from app.domain.entities import (
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    RiskFactor,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ─── Riesgo base por especialidad médica ─────────────────────────────────────
# Basado en literatura: especialidades con mayor incidencia de demandas
# Fuente: estadísticas de malpractice por especialidad

SPECIALTY_RISK_MAP: dict[str, float] = {
    # Alto riesgo (>0.6)
    "cirugía general": 0.75,
    "cirugía cardiovascular": 0.80,
    "cirugía plástica": 0.72,
    "neurocirugía": 0.82,
    "obstetricia": 0.78,
    "ginecología": 0.65,
    "anestesiología": 0.70,
    "traumatología": 0.65,
    "ortopedia": 0.63,
    # Riesgo medio (0.35–0.6)
    "cardiología": 0.55,
    "oncología": 0.50,
    "neurología": 0.48,
    "gastroenterología": 0.45,
    "urología": 0.50,
    "neumología": 0.42,
    "medicina interna": 0.40,
    "radiología": 0.45,
    "emergencias": 0.55,
    "medicina de emergencia": 0.55,
    # Riesgo bajo (<0.35)
    "pediatría": 0.35,
    "dermatología": 0.28,
    "oftalmología": 0.30,
    "psiquiatría": 0.32,
    "endocrinología": 0.30,
    "medicina familiar": 0.25,
    "rehabilitación": 0.20,
    "patología": 0.18,
    "medicina preventiva": 0.15,
}

DEFAULT_SPECIALTY_RISK = 0.40  # riesgo por defecto si no está mapeada


# ─── Factores de riesgo y sus pesos ─────────────────────────────────────────

FACTOR_WEIGHTS = {
    "specialty_risk": 0.25,           # Riesgo inherente de la especialidad
    "procedure_complexity": 0.20,     # Complejidad del procedimiento
    "documentation": 0.15,            # Documentación clínica
    "informed_consent": 0.15,         # Consentimiento informado
    "prior_complaints": 0.10,         # Quejas previas
    "time_factor": 0.08,              # Tiempo desde el incidente
    "priority": 0.07,                 # Prioridad asignada
}


def _normalize_specialty(specialty: str) -> str:
    """Normaliza el nombre de la especialidad para lookup."""
    return specialty.lower().strip()


def _get_specialty_risk(specialty: str) -> float:
    """Obtiene el riesgo base de una especialidad médica."""
    normalized = _normalize_specialty(specialty)
    # Búsqueda exacta
    if normalized in SPECIALTY_RISK_MAP:
        return SPECIALTY_RISK_MAP[normalized]
    # Búsqueda parcial
    for key, risk in SPECIALTY_RISK_MAP.items():
        if key in normalized or normalized in key:
            return risk
    return DEFAULT_SPECIALTY_RISK


def _complexity_value(complexity: str) -> float:
    """Convierte la complejidad del procedimiento a un valor numérico."""
    mapping = {"baja": 0.2, "media": 0.5, "alta": 0.85}
    return mapping.get(complexity.lower(), 0.5)


def _priority_value(priority: str) -> float:
    """Convierte la prioridad del caso a un valor numérico."""
    mapping = {"baja": 0.15, "media": 0.40, "alta": 0.70, "critica": 0.95}
    return mapping.get(priority.lower(), 0.40)


def _time_factor_value(days: int | None) -> float:
    """
    Calcula el factor de riesgo temporal.
    Casos recientes tienen mayor riesgo legal (evidencia fresca, urgencia).
    Casos muy viejos pueden estar fuera del plazo de prescripción.
    """
    if days is None:
        return 0.5  # desconocido → riesgo medio
    if days <= 7:
        return 0.90  # muy reciente → alto riesgo
    if days <= 30:
        return 0.75
    if days <= 90:
        return 0.60
    if days <= 365:
        return 0.40
    if days <= 730:
        return 0.25  # >1 año → riesgo decreciente
    return 0.10  # >2 años → posible prescripción


def _risk_level(score: float) -> str:
    """Clasifica el score de riesgo en un nivel legible."""
    if score >= 0.75:
        return "critico"
    if score >= 0.50:
        return "alto"
    if score >= 0.30:
        return "moderado"
    return "bajo"


def _generate_recommendations(
    request: RiskAssessmentRequest,
    risk_factors: list[RiskFactor],
    risk_score: float,
) -> list[str]:
    """Genera recomendaciones de mitigación basadas en los factores de riesgo."""
    recs: list[str] = []

    if not request.documentation_complete:
        recs.append(
            "URGENTE: Completar la documentación clínica del caso. "
            "La falta de registros es el factor más frecuente en demandas exitosas."
        )

    if not request.informed_consent:
        recs.append(
            "URGENTE: Obtener y documentar el consentimiento informado. "
            "Sin consentimiento válido, la defensa legal se debilita significativamente."
        )

    if request.has_prior_complaints:
        recs.append(
            "Se recomienda asignación prioritaria de abogado especializado "
            "dado el historial de quejas previas."
        )

    if request.procedure_complexity == "alta":
        recs.append(
            "Procedimiento de alta complejidad: asegurar que toda la cadena de "
            "decisiones médicas esté documentada con sus justificaciones."
        )

    spec_risk = _get_specialty_risk(request.specialty)
    if spec_risk >= 0.65:
        recs.append(
            f"La especialidad '{request.specialty}' tiene un perfil de riesgo alto. "
            "Se recomienda consulta legal preventiva para procedimientos programados."
        )

    if request.time_since_incident_days is not None:
        if request.time_since_incident_days <= 30:
            recs.append(
                "Incidente reciente: preservar toda la evidencia clínica, "
                "notas de evolución y resultados de exámenes sin modificaciones."
            )
        elif request.time_since_incident_days > 365:
            recs.append(
                "Han pasado más de 12 meses desde el incidente. "
                "Verificar plazos de prescripción según jurisdicción."
            )

    if risk_score >= 0.75:
        recs.append(
            "RIESGO CRÍTICO: Se recomienda intervención legal inmediata y "
            "notificación a la aseguradora de responsabilidad profesional."
        )
    elif risk_score >= 0.50:
        recs.append(
            "Riesgo alto: programar una evaluación legal detallada dentro de las "
            "próximas 48 horas."
        )

    if not recs:
        recs.append(
            "El caso presenta un perfil de riesgo bajo. Mantener la documentación "
            "actualizada y seguir protocolos estándar."
        )

    return recs


class RiskAssessmentModel:
    """
    Modelo de evaluación de riesgo médico-legal.

    Calcula un risk_score combinando factores ponderados y genera
    recomendaciones de mitigación.

    Cada factor produce:
        - Un valor normalizado (0-1)
        - Una contribución = peso × valor
        - Una descripción legible

    El risk_score final es la suma ponderada de todos los factores.
    """

    def __init__(self):
        self._version = settings.risk_model_version

    def assess(self, request: RiskAssessmentRequest) -> RiskAssessmentResponse:
        """Evalúa el riesgo de un caso médico-legal."""

        # Calcular cada factor
        specialty_risk = _get_specialty_risk(request.specialty)
        complexity_val = _complexity_value(request.procedure_complexity)
        doc_val = 0.0 if request.documentation_complete else 0.85
        consent_val = 0.0 if request.informed_consent else 0.90
        prior_val = 0.80 if request.has_prior_complaints else 0.10
        time_val = _time_factor_value(request.time_since_incident_days)
        priority_val = _priority_value(request.priority)

        # Construir factores con metadatos
        factors: list[RiskFactor] = [
            RiskFactor(
                name="Riesgo por especialidad",
                weight=FACTOR_WEIGHTS["specialty_risk"],
                value=round(specialty_risk, 4),
                contribution=round(FACTOR_WEIGHTS["specialty_risk"] * specialty_risk, 4),
                description=f"La especialidad '{request.specialty}' tiene un riesgo base de {specialty_risk:.0%}",
            ),
            RiskFactor(
                name="Complejidad del procedimiento",
                weight=FACTOR_WEIGHTS["procedure_complexity"],
                value=round(complexity_val, 4),
                contribution=round(FACTOR_WEIGHTS["procedure_complexity"] * complexity_val, 4),
                description=f"Procedimiento de complejidad {request.procedure_complexity}",
            ),
            RiskFactor(
                name="Documentación clínica",
                weight=FACTOR_WEIGHTS["documentation"],
                value=round(doc_val, 4),
                contribution=round(FACTOR_WEIGHTS["documentation"] * doc_val, 4),
                description=(
                    "Documentación completa"
                    if request.documentation_complete
                    else "ALERTA: Documentación incompleta — factor de riesgo crítico"
                ),
            ),
            RiskFactor(
                name="Consentimiento informado",
                weight=FACTOR_WEIGHTS["informed_consent"],
                value=round(consent_val, 4),
                contribution=round(FACTOR_WEIGHTS["informed_consent"] * consent_val, 4),
                description=(
                    "Consentimiento informado obtenido"
                    if request.informed_consent
                    else "ALERTA: Sin consentimiento informado — riesgo legal elevado"
                ),
            ),
            RiskFactor(
                name="Historial de quejas",
                weight=FACTOR_WEIGHTS["prior_complaints"],
                value=round(prior_val, 4),
                contribution=round(FACTOR_WEIGHTS["prior_complaints"] * prior_val, 4),
                description=(
                    "Existen quejas previas registradas"
                    if request.has_prior_complaints
                    else "Sin quejas previas"
                ),
            ),
            RiskFactor(
                name="Factor temporal",
                weight=FACTOR_WEIGHTS["time_factor"],
                value=round(time_val, 4),
                contribution=round(FACTOR_WEIGHTS["time_factor"] * time_val, 4),
                description=(
                    f"Incidente hace {request.time_since_incident_days} días"
                    if request.time_since_incident_days is not None
                    else "Tiempo desde el incidente desconocido"
                ),
            ),
            RiskFactor(
                name="Prioridad del caso",
                weight=FACTOR_WEIGHTS["priority"],
                value=round(priority_val, 4),
                contribution=round(FACTOR_WEIGHTS["priority"] * priority_val, 4),
                description=f"Prioridad asignada: {request.priority}",
            ),
        ]

        # Score total = suma ponderada de contribuciones
        risk_score = sum(f.contribution for f in factors)
        risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

        # Generar recomendaciones
        recommendations = _generate_recommendations(request, factors, risk_score)

        return RiskAssessmentResponse(
            case_id=request.case_id,
            risk_score=risk_score,
            risk_level=_risk_level(risk_score),
            risk_factors=factors,
            recommendations=recommendations,
            specialty_risk_baseline=round(specialty_risk, 4),
            model_version=self._version,
        )


# Singleton
risk_model = RiskAssessmentModel()
