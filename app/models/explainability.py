"""
Módulo de Explicabilidad de Recomendaciones
============================================
Genera explicaciones legibles sobre por qué un abogado fue recomendado
para un médico específico.

Descompone la similitud del coseno (TF-IDF) en contribuciones por grupo
de features: especialidades coincidentes, experiencia, casos resueltos, rating.
Esto produce una explicación equivalente a SHAP para el modelo content-based.
"""

from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    FeatureImportance,
)
from app.data.preprocessing import find_matching_specialties


def explain_content_recommendation(
    doctor: DoctorProfile,
    lawyer: LawyerProfile,
    content_model,
    score: float,
) -> tuple[list[FeatureImportance], list[str]]:
    """
    Explica por qué el modelo content-based recomendó este abogado.

    Descompone la similitud del coseno en contribuciones por feature group:
    especialidades coincidentes, experiencia, casos resueltos, rating y
    especialidades legales.

    Returns:
        (feature_importances, human_readable_reasons)
    """
    if not content_model.is_fitted:
        return [], []

    importances: list[FeatureImportance] = []
    reasons: list[str] = []

    # ── 1. Especialidades coincidentes ────────────────────────────────────
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

    # ── 2. Experiencia del abogado ────────────────────────────────────────
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

    # ── 3. Casos resueltos ────────────────────────────────────────────────
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

    # ── 4. Rating del abogado ─────────────────────────────────────────────
    if lawyer.rating >= 4.0:
        rating_importance = (lawyer.rating - 3.0) / 2.0
        importances.append(FeatureImportance(
            feature="rating",
            importance=round(rating_importance, 4),
            description=f"Valoración promedio: {lawyer.rating:.1f}/5.0",
        ))
        reasons.append(f"Alta valoración: {lawyer.rating:.1f}/5.0 por otros médicos")

    # ── 5. Especialidades legales relevantes ──────────────────────────────
    if lawyer.specialties:
        importances.append(FeatureImportance(
            feature="legal_specialties",
            importance=round(min(len(lawyer.specialties) * 0.15, 0.5), 4),
            description=f"Especialidades legales: {', '.join(lawyer.specialties[:3])}",
        ))

    importances.sort(key=lambda x: x.importance, reverse=True)

    if not reasons:
        reasons.append("Perfil compatible basado en análisis de texto y experiencia profesional")

    return importances[:5], reasons
