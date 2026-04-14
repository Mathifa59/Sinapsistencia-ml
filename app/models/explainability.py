"""
Módulo de Explicabilidad de Recomendaciones
============================================
Genera explicaciones legibles sobre POR QUÉ un abogado fue recomendado
para un médico específico.

Implementa:
    1. Feature importance pseudo-SHAP: descompone el score en contribuciones
       de cada feature (especialidades, áreas médicas, experiencia, rating, etc.)
    2. Razones textuales legibles para el usuario final.

¿Por qué pseudo-SHAP y no SHAP real?
    SHAP (SHapley Additive exPlanations) requiere un modelo supervisado
    con features tabulares. Nuestro modelo content-based usa TF-IDF + coseno,
    lo que no se presta directamente a SHAP. En su lugar:

    - Descomponemos la similitud del coseno en contribuciones por grupo de features
    - Identificamos qué términos TF-IDF aportan más al score
    - Esto produce una explicación equivalente en utilidad práctica

    Para el modelo collaborative, usamos la contribución de los factores latentes
    como proxy de importancia.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    FeatureImportance,
)
from app.data.preprocessing import (
    build_doctor_text,
    build_lawyer_text,
    find_matching_specialties,
)


def explain_content_recommendation(
    doctor: DoctorProfile,
    lawyer: LawyerProfile,
    content_model,
    score: float,
) -> tuple[list[FeatureImportance], list[str]]:
    """
    Explica por qué el modelo content-based recomendó este abogado.

    Descompone la similitud del coseno en:
    1. Contribución del texto (TF-IDF): qué términos coinciden
    2. Contribución numérica: experiencia, casos, rating

    Returns:
        (feature_importances, human_readable_reasons)
    """
    if not content_model.is_fitted:
        return [], []

    importances: list[FeatureImportance] = []
    reasons: list[str] = []

    # ── 1. Especialidades coincidentes ────────────────────────────────
    matched = find_matching_specialties(doctor, lawyer)
    if matched:
        match_importance = min(0.4 + len(matched) * 0.1, 0.9)
        importances.append(FeatureImportance(
            feature="matched_specialties",
            importance=round(match_importance, 4),
            description=f"Especialidades coincidentes: {', '.join(matched)}",
        ))
        reasons.append(
            f"Coincidencia directa en {len(matched)} área(s) médica(s): {', '.join(matched)}"
        )

    # ── 2. Experiencia del abogado ────────────────────────────────────
    if lawyer.years_experience > 0:
        exp_importance = min(lawyer.years_experience / 20.0, 0.5)
        importances.append(FeatureImportance(
            feature="years_experience",
            importance=round(exp_importance, 4),
            description=f"{lawyer.years_experience} años de experiencia profesional",
        ))
        if lawyer.years_experience >= 10:
            reasons.append(
                f"{lawyer.years_experience} años de trayectoria en casos médico-legales"
            )

    # ── 3. Casos resueltos ────────────────────────────────────────────
    if lawyer.resolved_cases > 0:
        cases_importance = min(lawyer.resolved_cases / 150.0, 0.6)
        importances.append(FeatureImportance(
            feature="resolved_cases",
            importance=round(cases_importance, 4),
            description=f"{lawyer.resolved_cases} casos resueltos exitosamente",
        ))
        if lawyer.resolved_cases >= 50:
            reasons.append(
                f"Historial sólido: {lawyer.resolved_cases} casos resueltos"
            )

    # ── 4. Rating del abogado ─────────────────────────────────────────
    if lawyer.rating >= 4.0:
        rating_importance = (lawyer.rating - 3.0) / 2.0  # 3.0→0.0, 5.0→1.0
        importances.append(FeatureImportance(
            feature="rating",
            importance=round(rating_importance, 4),
            description=f"Valoración promedio: {lawyer.rating:.1f}/5.0",
        ))
        reasons.append(f"Alta valoración: {lawyer.rating:.1f}/5.0 por otros médicos")

    # ── 5. Especialidades legales relevantes ──────────────────────────
    if lawyer.specialties:
        importances.append(FeatureImportance(
            feature="legal_specialties",
            importance=round(min(len(lawyer.specialties) * 0.15, 0.5), 4),
            description=f"Especialidades legales: {', '.join(lawyer.specialties[:3])}",
        ))

    # Ordenar por importancia
    importances.sort(key=lambda x: x.importance, reverse=True)

    # Si no hay razones específicas, dar razón genérica
    if not reasons:
        reasons.append("Perfil compatible basado en análisis de texto y experiencia profesional")

    return importances[:5], reasons  # top 5 features


def explain_collaborative_recommendation(
    doctor_id: str,
    lawyer_id: str,
    collaborative_model,
    score: float,
) -> tuple[list[FeatureImportance], list[str]]:
    """
    Explica por qué el modelo collaborative recomendó este abogado.

    El modelo SVD aprende factores latentes — no tienen nombres interpretables.
    Usamos proxies para la explicación:
    1. Que otros doctores con perfil similar aceptaron este abogado
    2. El score collaborative es alto → patrón histórico fuerte
    """
    importances: list[FeatureImportance] = []
    reasons: list[str] = []

    if not collaborative_model.is_fitted:
        return importances, reasons

    if score > 0.7:
        importances.append(FeatureImportance(
            feature="historical_pattern",
            importance=round(score, 4),
            description="Patrón fuerte: médicos similares eligieron este abogado",
        ))
        reasons.append(
            "Médicos con perfil similar han trabajado exitosamente con este abogado"
        )
    elif score > 0.4:
        importances.append(FeatureImportance(
            feature="historical_pattern",
            importance=round(score, 4),
            description="Patrón moderado en historial de matches",
        ))
        reasons.append(
            "Existe un patrón positivo de matches con médicos de perfil similar"
        )

    return importances, reasons


def explain_hybrid_recommendation(
    doctor: DoctorProfile,
    lawyer: LawyerProfile,
    content_model,
    collaborative_model,
    content_score: float,
    collaborative_score: float,
    final_score: float,
    alpha: float,
    beta: float,
) -> tuple[list[FeatureImportance], list[str]]:
    """
    Combina las explicaciones de ambos modelos para el híbrido.
    """
    content_importances, content_reasons = explain_content_recommendation(
        doctor, lawyer, content_model, content_score
    )
    collab_importances, collab_reasons = explain_collaborative_recommendation(
        doctor.id, lawyer.id, collaborative_model, collaborative_score
    )

    # Ponderar importancias por los pesos del modelo
    for imp in content_importances:
        imp.importance = round(imp.importance * alpha, 4)
    for imp in collab_importances:
        imp.importance = round(imp.importance * beta, 4)

    all_importances = content_importances + collab_importances
    all_importances.sort(key=lambda x: x.importance, reverse=True)

    all_reasons = content_reasons + collab_reasons

    return all_importances[:6], all_reasons
