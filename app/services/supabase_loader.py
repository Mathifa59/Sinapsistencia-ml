"""
Supabase Data Loader — Carga datos reales para entrenamiento
=============================================================
Lee perfiles de abogados e interacciones directamente desde Supabase
para entrenar los modelos con datos reales (no de muestra).

Esto resuelve el problema de que los IDs de muestra (law_001, doc_001)
no coinciden con los UUIDs reales de Supabase.
"""

import logging
from datetime import datetime, timezone

import httpx

from app.domain.entities import LawyerProfile, Interaction

logger = logging.getLogger(__name__)


async def load_lawyers_from_supabase(
    supabase_url: str,
    supabase_key: str,
) -> list[LawyerProfile]:
    """
    Carga perfiles de abogados desde Supabase.

    Query: lawyer_profiles JOIN profiles para obtener nombre.
    Los IDs que se usan son los user_id (UUID) de Supabase,
    que coinciden con los que envía el frontend.
    """
    url = f"{supabase_url}/rest/v1/lawyer_profiles"
    params = {
        "select": "*, user:profiles!lawyer_profiles_user_id_fkey(id, name)",
    }
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

    lawyers: list[LawyerProfile] = []
    for row in data:
        user = row.get("user") or {}
        lawyers.append(LawyerProfile(
            id=row["user_id"],  # UUID real — matchea con doctor_id del frontend
            name=user.get("name", ""),
            specialties=row.get("specialties") or [],
            medical_areas=row.get("medical_areas") or [],
            years_experience=row.get("years_experience") or 0,
            resolved_cases=row.get("resolved_cases") or 0,
            rating=float(row.get("rating") or 0),
            description=row.get("bio") or "",
        ))

    logger.info("Cargados %d abogados desde Supabase", len(lawyers))
    return lawyers


async def load_interactions_from_supabase(
    supabase_url: str,
    supabase_key: str,
) -> list[Interaction]:
    """
    Carga interacciones doctor-abogado desde la tabla contact_requests.

    Mapeo de estados a interacciones:
        - "aceptado" → accepted=True, rating basado en ml_score si existe
        - "rechazado" → accepted=False
        - "pendiente"/"cancelado" → se ignoran (no son señal clara)

    Esto permite al modelo collaborative learning aprender de las
    decisiones reales de aceptación/rechazo entre médicos y abogados.
    """
    url = f"{supabase_url}/rest/v1/contact_requests"
    params = {
        "select": "from_doctor_id, to_lawyer_id, status, ml_score, created_at",
        "status": "in.(aceptado,rechazado)",
        "order": "created_at.asc",
    }
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

    interactions: list[Interaction] = []
    for row in data:
        status = row.get("status", "")
        if status not in ("aceptado", "rechazado"):
            continue

        accepted = status == "aceptado"
        rating = None
        if row.get("ml_score") and accepted:
            # Convertir ml_score (0-100) a rating (1-5)
            ml_score = float(row["ml_score"])
            rating = 1.0 + (ml_score / 100.0) * 4.0
            rating = round(min(max(rating, 1.0), 5.0), 2)

        timestamp = datetime.fromisoformat(
            row.get("created_at", datetime.now(timezone.utc).isoformat())
        )

        interactions.append(Interaction(
            doctor_id=row["from_doctor_id"],
            lawyer_id=row["to_lawyer_id"],
            accepted=accepted,
            rating=rating,
            timestamp=timestamp,
        ))

    logger.info(
        "Cargadas %d interacciones desde Supabase (%d aceptadas, %d rechazadas)",
        len(interactions),
        sum(1 for i in interactions if i.accepted),
        sum(1 for i in interactions if not i.accepted),
    )
    return interactions
