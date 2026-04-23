"""
Modelo: Evaluación de Riesgo Médico-Legal con Random Forest
===========================================================
Evalúa el riesgo legal de un caso médico combinando:
    1. Factores ponderados (reglas basadas en literatura médico-legal)
    2. Random Forest entrenado sobre datos sintéticos generados desde las reglas

Algoritmo — Random Forest (seleccionado por benchmarking, puntaje 3.49):
    Ventajas que justifican la elección:
    - Alto rendimiento sin sobreajuste (ensamble de árboles)
    - Robustez ante datos ruidosos/incompletos (común en registros clínico-legales)
    - Interpretabilidad via feature importances (explica qué factores pesan más)
    - Velocidad adecuada para entornos con infraestructura limitada

Flujo de predicción:
    1. Se extraen 7 features numéricas del caso (riesgo especialidad, complejidad,
       documentación, consentimiento, quejas previas, factor temporal, prioridad).
    2. El RF predice probabilidades de cada clase de riesgo [bajo, moderado, alto, crítico].
    3. El risk_score continuo se calcula como suma ponderada de probabilidades
       por el centro de cada clase.
    4. Se generan recomendaciones de mitigación basadas en los factores individuales.

Entrenamiento inicial:
    Con pocos datos reales disponibles, el RF se inicializa con datos sintéticos
    generados muestreando el espacio de features y etiquetando con las reglas.
    Cuando haya datos reales, basta llamar model.fit(X_real, y_real).

Épica 13 del Product Backlog:
    HU-044: Evaluación automática de riesgo de un caso
    HU-045: Factores de riesgo desglosados con pesos
    HU-046: Recomendaciones de mitigación
    HU-047: Historial de evaluaciones de riesgo
"""

import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.domain.entities import (
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    RiskFactor,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ─── Riesgo base por especialidad médica ─────────────────────────────────────

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

DEFAULT_SPECIALTY_RISK = 0.40


# ─── Factores de riesgo y sus pesos ──────────────────────────────────────────
# El orden de este dict define el orden del vector de features del RF.

FACTOR_WEIGHTS = {
    "specialty_risk":       0.25,
    "procedure_complexity": 0.20,
    "documentation":        0.15,
    "informed_consent":     0.15,
    "prior_complaints":     0.10,
    "time_factor":          0.08,
    "priority":             0.07,
}

FEATURE_NAMES = list(FACTOR_WEIGHTS.keys())
FEATURE_WEIGHTS_ARRAY = np.array(list(FACTOR_WEIGHTS.values()))

# Centros de cada clase de riesgo — usados para convertir probabilidades a score continuo.
# bajo=[0, 0.30], moderado=[0.30, 0.50], alto=[0.50, 0.75], critico=[0.75, 1.0]
CLASS_CENTERS = np.array([0.15, 0.40, 0.625, 0.875])
RISK_CLASSES = ["bajo", "moderado", "alto", "critico"]


# ─── Helpers de features ─────────────────────────────────────────────────────

def _normalize_specialty(specialty: str) -> str:
    return specialty.lower().strip()


def _get_specialty_risk(specialty: str) -> float:
    normalized = _normalize_specialty(specialty)
    if normalized in SPECIALTY_RISK_MAP:
        return SPECIALTY_RISK_MAP[normalized]
    for key, risk in SPECIALTY_RISK_MAP.items():
        if key in normalized or normalized in key:
            return risk
    return DEFAULT_SPECIALTY_RISK


def _complexity_value(complexity: str) -> float:
    return {"baja": 0.2, "media": 0.5, "alta": 0.85}.get(complexity.lower(), 0.5)


def _priority_value(priority: str) -> float:
    return {"baja": 0.15, "media": 0.40, "alta": 0.70, "critica": 0.95}.get(
        priority.lower(), 0.40
    )


def _time_factor_value(days: int | None) -> float:
    if days is None:
        return 0.5
    if days <= 7:
        return 0.90
    if days <= 30:
        return 0.75
    if days <= 90:
        return 0.60
    if days <= 365:
        return 0.40
    if days <= 730:
        return 0.25
    return 0.10


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "critico"
    if score >= 0.50:
        return "alto"
    if score >= 0.30:
        return "moderado"
    return "bajo"


def _extract_feature_vector(request: RiskAssessmentRequest) -> np.ndarray:
    """Convierte un RiskAssessmentRequest en el vector de 7 features del RF."""
    return np.array([[
        _get_specialty_risk(request.specialty),
        _complexity_value(request.procedure_complexity),
        0.0 if request.documentation_complete else 0.85,
        0.0 if request.informed_consent else 0.90,
        0.80 if request.has_prior_complaints else 0.10,
        _time_factor_value(request.time_since_incident_days),
        _priority_value(request.priority),
    ]])


# ─── Generación de datos sintéticos ──────────────────────────────────────────

def _generate_synthetic_training_data(
    n_samples: int = 3000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera un dataset sintético para el entrenamiento inicial del RF.

    Estrategia: muestrea combinaciones aleatorias del espacio de 7 features
    y las etiqueta con el modelo de reglas (score ponderado → clase de riesgo).
    Esto bootstrappea el RF desde conocimiento experto, sin necesidad de datos
    históricos reales. Cuando existan datos reales, se reemplaza este dataset.
    """
    rng = np.random.default_rng(random_state)

    specialty_pool = list(SPECIALTY_RISK_MAP.values()) + [DEFAULT_SPECIALTY_RISK]

    specialty_risks  = rng.choice(specialty_pool, n_samples)
    complexity_vals  = rng.choice([0.2, 0.5, 0.85], n_samples)
    doc_vals         = rng.choice([0.0, 0.85], n_samples, p=[0.55, 0.45])
    consent_vals     = rng.choice([0.0, 0.90], n_samples, p=[0.65, 0.35])
    prior_vals       = rng.choice([0.10, 0.80], n_samples, p=[0.75, 0.25])
    time_vals        = rng.uniform(0.0, 1.0, n_samples)
    priority_vals    = rng.choice([0.15, 0.40, 0.70, 0.95], n_samples)

    X = np.column_stack([
        specialty_risks, complexity_vals, doc_vals, consent_vals,
        prior_vals, time_vals, priority_vals,
    ])

    # Etiquetas: score ponderado por las reglas → clase discreta
    scores = np.clip(X @ FEATURE_WEIGHTS_ARRAY, 0.0, 1.0)
    y = np.where(scores >= 0.75, 3,
        np.where(scores >= 0.50, 2,
        np.where(scores >= 0.30, 1, 0)))

    return X, y


# ─── Recomendaciones de mitigación ───────────────────────────────────────────

def _generate_recommendations(
    request: RiskAssessmentRequest,
    risk_factors: list[RiskFactor],
    risk_score: float,
) -> list[str]:
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


# ─── Modelo ───────────────────────────────────────────────────────────────────

class RiskAssessmentModel:
    """
    Modelo de evaluación de riesgo médico-legal basado en Random Forest.

    El RF es entrenado inicialmente con datos sintéticos generados desde las
    reglas expertas. Puede re-entrenarse con datos reales mediante fit(X, y).

    Cada predicción produce:
        - risk_score (0-1): suma ponderada de probabilidades de clase RF
        - risk_level: bajo / moderado / alto / critico
        - risk_factors: desglose de cada factor con su peso y contribución
        - recommendations: acciones de mitigación concretas
        - feature_importances: qué features pesan más en el RF (XAI)
    """

    RF_PARAMS = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "random_state": 42,
        "class_weight": "balanced",
    }

    def __init__(self):
        self._version = settings.risk_model_version
        self._rf: RandomForestClassifier | None = None
        self._is_fitted: bool = False
        # Inicializa el RF con datos sintéticos para que esté listo desde el arranque
        self.fit()

    # ─── Entrenamiento ─────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "RiskAssessmentModel":
        """
        Entrena el Random Forest.

        Args:
            X: Matriz (n_muestras × 7 features). Si None, genera datos sintéticos.
            y: Vector de etiquetas (0=bajo, 1=moderado, 2=alto, 3=critico).
               Si None, se infiere junto a X desde el modelo de reglas.

        Returns:
            self
        """
        if X is None or y is None:
            logger.info("Entrenando Risk RF con datos sintéticos (%d muestras)...", 3000)
            X, y = _generate_synthetic_training_data()

        self._rf = RandomForestClassifier(**self.RF_PARAMS)
        self._rf.fit(X, y)
        self._is_fitted = True
        logger.info("Risk RF entrenado. Clases: %s", self._rf.classes_.tolist())
        return self

    # ─── Predicción ────────────────────────────────────────────────────────

    def assess(self, request: RiskAssessmentRequest) -> RiskAssessmentResponse:
        """Evalúa el riesgo de un caso médico-legal."""

        # ── Calcular valores de cada factor ──────────────────────────────
        specialty_risk = _get_specialty_risk(request.specialty)
        complexity_val = _complexity_value(request.procedure_complexity)
        doc_val        = 0.0 if request.documentation_complete else 0.85
        consent_val    = 0.0 if request.informed_consent else 0.90
        prior_val      = 0.80 if request.has_prior_complaints else 0.10
        time_val       = _time_factor_value(request.time_since_incident_days)
        priority_val   = _priority_value(request.priority)

        factor_values = [
            specialty_risk, complexity_val, doc_val,
            consent_val, prior_val, time_val, priority_val,
        ]

        # ── Construir lista de factores con metadatos ─────────────────────
        factor_meta = [
            (
                "Riesgo por especialidad",
                f"La especialidad '{request.specialty}' tiene un riesgo base de {specialty_risk:.0%}",
            ),
            (
                "Complejidad del procedimiento",
                f"Procedimiento de complejidad {request.procedure_complexity}",
            ),
            (
                "Documentación clínica",
                "Documentación completa"
                if request.documentation_complete
                else "ALERTA: Documentación incompleta — factor de riesgo crítico",
            ),
            (
                "Consentimiento informado",
                "Consentimiento informado obtenido"
                if request.informed_consent
                else "ALERTA: Sin consentimiento informado — riesgo legal elevado",
            ),
            (
                "Historial de quejas",
                "Existen quejas previas registradas"
                if request.has_prior_complaints
                else "Sin quejas previas",
            ),
            (
                "Factor temporal",
                f"Incidente hace {request.time_since_incident_days} días"
                if request.time_since_incident_days is not None
                else "Tiempo desde el incidente desconocido",
            ),
            (
                "Prioridad del caso",
                f"Prioridad asignada: {request.priority}",
            ),
        ]

        factors: list[RiskFactor] = [
            RiskFactor(
                name=name,
                weight=FACTOR_WEIGHTS[key],
                value=round(val, 4),
                contribution=round(FACTOR_WEIGHTS[key] * val, 4),
                description=desc,
            )
            for (name, desc), key, val in zip(
                factor_meta, FEATURE_NAMES, factor_values
            )
        ]

        # ── Random Forest: predice probabilidades de cada clase ───────────
        features = _extract_feature_vector(request)
        class_probs = self._rf.predict_proba(features)[0]

        # Score continuo = suma ponderada por centros de clase
        # Ejemplo: [0.1, 0.3, 0.5, 0.1] · [0.15, 0.40, 0.625, 0.875] = 0.51
        risk_score = float(np.dot(class_probs, CLASS_CENTERS))
        risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

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

    # ─── Feature importance (XAI) ─────────────────────────────────────────

    def get_feature_importances(self) -> dict[str, float]:
        """
        Retorna la importancia de cada feature aprendida por el RF.
        Permite explicar qué factores pesan más en la predicción de riesgo.
        """
        if not self._is_fitted:
            return {}
        return {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, self._rf.feature_importances_)
        }

    # ─── Serialización ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serializa el RF entrenado a disco."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self._rf, os.path.join(path, "risk_rf.pkl"))
        logger.info("Risk RF guardado en '%s/risk_rf.pkl'.", path)

    def load(self, path: str) -> "RiskAssessmentModel":
        """Carga un RF previamente serializado. Si no existe, mantiene el sintético."""
        rf_path = os.path.join(path, "risk_rf.pkl")
        if os.path.exists(rf_path):
            self._rf = joblib.load(rf_path)
            self._is_fitted = True
            logger.info("Risk RF cargado desde '%s'.", rf_path)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# Singleton
risk_model = RiskAssessmentModel()
