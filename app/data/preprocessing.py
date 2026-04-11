"""
Feature Engineering — Preprocesamiento de perfiles médico-legales
=================================================================
Transforma los perfiles de médicos y abogados en vectores numéricos
que los modelos de ML pueden procesar.

Pipeline:
    1. Construcción del texto de perfil   → concatena campos relevantes
    2. Vectorización TF-IDF               → representa texto como vector numérico
    3. Features numéricas                 → normalización Min-Max
    4. Vector final                       → concatenación de ambos

¿Por qué TF-IDF?
    Term Frequency-Inverse Document Frequency pondera las palabras según
    su frecuencia en el perfil (TF) versus su rareza en todos los perfiles (IDF).
    Eso hace que especialidades específicas como "Responsabilidad Civil Médica
    Cardiológica" tengan más peso que términos genéricos como "derecho".
"""

import numpy as np
from typing import Sequence

from app.domain.entities import DoctorProfile, LawyerProfile


# ─── Construcción del texto de perfil ─────────────────────────────────────────


def build_lawyer_text(lawyer: LawyerProfile) -> str:
    """
    Construye el texto representativo del perfil de un abogado.
    Este texto es el 'documento' que TF-IDF vectorizará.

    Estrategia de ponderación manual:
    - Las especialidades se repiten 3 veces → más peso en TF-IDF
    - Las áreas médicas se repiten 2 veces → peso intermedio
    - La descripción libre se incluye una vez
    """
    parts: list[str] = []

    # Especialidades legales: mayor peso (x3)
    for specialty in lawyer.specialties:
        parts.extend([specialty.lower()] * 3)

    # Áreas médicas que domina: peso intermedio (x2)
    for area in lawyer.medical_areas:
        parts.extend([area.lower()] * 2)

    # Texto libre del perfil (x1)
    if lawyer.description:
        parts.append(lawyer.description.lower())

    return " ".join(parts)


def build_doctor_text(doctor: DoctorProfile) -> str:
    """
    Construye el texto de consulta del médico.
    Representa qué tipo de abogado necesita.

    La especialidad principal tiene mayor peso porque define
    el área médica del caso que necesita asesoría legal.
    """
    parts: list[str] = []

    # Especialidad principal: mayor peso (x4)
    parts.extend([doctor.specialty.lower()] * 4)

    # Sub-especialidades: peso intermedio (x2)
    for sub in doctor.sub_specialties:
        parts.extend([sub.lower()] * 2)

    # Hospital/contexto (x1)
    if doctor.hospital:
        parts.append(doctor.hospital.lower())

    # Descripción libre (x1)
    if doctor.description:
        parts.append(doctor.description.lower())

    return " ".join(parts)


# ─── Features numéricas ───────────────────────────────────────────────────────


def extract_lawyer_numerical_features(lawyer: LawyerProfile) -> np.ndarray:
    """
    Extrae las features numéricas de un abogado como vector sin normalizar.

    Features:
        [0] años_experiencia  (0 – ~40)
        [1] casos_resueltos   (0 – ~500)
        [2] valoracion        (0 – 5)
    """
    return np.array([
        float(lawyer.years_experience),
        float(lawyer.resolved_cases),
        float(lawyer.rating),
    ])


def extract_doctor_numerical_features(doctor: DoctorProfile) -> np.ndarray:
    """
    Extrae las features numéricas de un médico como vector sin normalizar.

    Features:
        [0] años_experiencia  (0 – ~40)
    """
    return np.array([
        float(doctor.years_experience),
    ])


# ─── Especialidades coincidentes ──────────────────────────────────────────────


def find_matching_specialties(
    doctor: DoctorProfile,
    lawyer: LawyerProfile,
) -> list[str]:
    """
    Detecta qué especialidades médicas tiene el abogado que coinciden
    con la especialidad/sub-especialidades del médico.
    Usado para explicar la recomendación al usuario.
    """
    doctor_specialties = {doctor.specialty.lower()} | {
        s.lower() for s in doctor.sub_specialties
    }
    lawyer_medical = {a.lower() for a in lawyer.medical_areas}

    matched = doctor_specialties & lawyer_medical

    # También busca coincidencias parciales (ej: "cardio" en "cardiología")
    for doc_spec in doctor_specialties:
        for law_area in lawyer_medical:
            if doc_spec in law_area or law_area in doc_spec:
                matched.add(law_area)

    return sorted(matched)


# ─── Validación de datos de entrada ───────────────────────────────────────────


def validate_lawyer_profiles(lawyers: Sequence[LawyerProfile]) -> list[str]:
    """
    Valida que los perfiles de abogados tengan los campos mínimos necesarios.
    Retorna una lista de warnings (vacía si todo está bien).
    """
    warnings: list[str] = []
    for lawyer in lawyers:
        if not lawyer.specialties:
            warnings.append(f"Abogado {lawyer.id} no tiene especialidades definidas")
        if not lawyer.medical_areas:
            warnings.append(
                f"Abogado {lawyer.id} no tiene áreas médicas definidas — "
                "las recomendaciones serán menos precisas"
            )
    return warnings
