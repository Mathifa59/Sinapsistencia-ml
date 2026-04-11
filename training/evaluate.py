"""
Métricas de evaluación del sistema de recomendación
====================================================
Implementa las métricas estándar para evaluar la calidad del ranking.

Métricas implementadas:
    - Precision@K  : ¿Cuántos de los top-K recomendados son relevantes?
    - Recall@K     : ¿Qué fracción de los relevantes aparecen en el top-K?
    - NDCG@K       : Normalized Discounted Cumulative Gain — mide calidad del ranking
    - MAP          : Mean Average Precision — promedio de precisión por doctor

Estas métricas se usan en la literatura de sistemas de recomendación
(ej: RecSys, Netflix Prize) y son las que se deben reportar en la tesis.

Protocolo de evaluación:
    - Split temporal: 80% train, 20% test (más realista que split aleatorio)
    - Para cada doctor en test: las recomendaciones generadas vs. las reales
    - "Relevante" = abogado que el médico aceptó en el set de test
"""

import numpy as np
from dataclasses import dataclass, field

from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    Interaction,
    RecommendationScore,
    ModelMetrics,
)
from app.models.hybrid import HybridRecommender


# ─── Métricas individuales ────────────────────────────────────────────────────


def precision_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Precision@K = |recomendados_top_k ∩ relevantes| / K

    ¿De los K recomendados, cuántos son realmente relevantes?
    Penaliza incluir items irrelevantes en el top-K.
    """
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Recall@K = |recomendados_top_k ∩ relevantes| / |relevantes|

    ¿Qué fracción de todos los relevantes encontró el sistema en el top-K?
    Penaliza no encontrar items relevantes.
    """
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Discounted Cumulative Gain @K.

    Suma las ganancias con descuento logarítmico según la posición:
        DCG@K = Σ rel_i / log2(i + 1)   para i en [1, K]

    Un item relevante en posición 1 vale más que en posición 10.
    """
    dcg = 0.0
    for i, rid in enumerate(recommended_ids[:k], start=1):
        if rid in relevant_ids:
            dcg += 1.0 / np.log2(i + 1)
    return dcg


def ndcg_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Normalized DCG @K = DCG@K / IDCG@K

    IDCG (Ideal DCG) es el DCG máximo posible si el ranking fuera perfecto.
    Normaliza DCG a [0, 1]: 1.0 = ranking perfecto.
    """
    dcg = dcg_at_k(recommended_ids, relevant_ids, k)
    # IDCG: todos los relevantes en las primeras posiciones
    ideal_ranking = list(relevant_ids)[:k]
    idcg = dcg_at_k(ideal_ranking, relevant_ids, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(recommended_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Average Precision para un doctor.
    Promedia la precision en cada posición donde hay un hit relevante.
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(recommended_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant_ids)


# ─── Evaluación completa del sistema ─────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Resultado completo de una evaluación."""
    model_name: str
    k: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map_score: float
    num_doctors_evaluated: int
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  Evaluación: {self.model_name} @ K={self.k}\n"
            f"{'='*55}\n"
            f"  Precision@{self.k}  : {self.precision_at_k:.4f}\n"
            f"  Recall@{self.k}     : {self.recall_at_k:.4f}\n"
            f"  NDCG@{self.k}       : {self.ndcg_at_k:.4f}\n"
            f"  MAP         : {self.map_score:.4f}\n"
            f"  Doctores eval: {self.num_doctors_evaluated}\n"
            f"{'='*55}\n"
        )


def evaluate_model(
    model: HybridRecommender,
    test_doctors: list[DoctorProfile],
    test_interactions: list[Interaction],
    k: int = 10,
    model_name: str = "hybrid",
) -> EvaluationResult:
    """
    Evalúa el modelo sobre un conjunto de test.

    Protocolo:
        Para cada doctor en test_doctors:
            1. Obtener los abogados que aceptó en test_interactions (ground truth)
            2. Generar las top-K recomendaciones del modelo
            3. Calcular Precision@K, Recall@K, NDCG@K, AP
        Promediar todas las métricas.

    Args:
        model           : HybridRecommender ya entrenado con datos de train.
        test_doctors    : Médicos del set de test.
        test_interactions: Interacciones positivas (accepted=True) del set de test.
        k               : Número de recomendaciones a evaluar.
        model_name      : Nombre para el reporte.

    Returns:
        EvaluationResult con todas las métricas.
    """
    # Ground truth: para cada doctor, qué abogados aceptó en el test
    ground_truth: dict[str, set[str]] = {}
    for interaction in test_interactions:
        if interaction.accepted:
            if interaction.doctor_id not in ground_truth:
                ground_truth[interaction.doctor_id] = set()
            ground_truth[interaction.doctor_id].add(interaction.lawyer_id)

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    aps: list[float] = []
    details: list[dict] = []

    doctors_evaluated = 0
    for doctor in test_doctors:
        relevant_ids = ground_truth.get(doctor.id, set())
        if not relevant_ids:
            continue  # sin ground truth no se puede evaluar este doctor

        # Genera recomendaciones
        recommendations = model.recommend(doctor, top_k=k)
        recommended_ids = [r.lawyer_id for r in recommendations]

        # Calcula métricas
        p_k = precision_at_k(recommended_ids, relevant_ids, k)
        r_k = recall_at_k(recommended_ids, relevant_ids, k)
        n_k = ndcg_at_k(recommended_ids, relevant_ids, k)
        ap = average_precision(recommended_ids, relevant_ids)

        precisions.append(p_k)
        recalls.append(r_k)
        ndcgs.append(n_k)
        aps.append(ap)
        doctors_evaluated += 1

        details.append({
            "doctor_id": doctor.id,
            "doctor_specialty": doctor.specialty,
            "relevant_count": len(relevant_ids),
            "precision_at_k": round(p_k, 4),
            "recall_at_k": round(r_k, 4),
            "ndcg_at_k": round(n_k, 4),
            "average_precision": round(ap, 4),
        })

    if doctors_evaluated == 0:
        return EvaluationResult(
            model_name=model_name,
            k=k,
            precision_at_k=0.0,
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            map_score=0.0,
            num_doctors_evaluated=0,
        )

    return EvaluationResult(
        model_name=model_name,
        k=k,
        precision_at_k=float(np.mean(precisions)),
        recall_at_k=float(np.mean(recalls)),
        ndcg_at_k=float(np.mean(ndcgs)),
        map_score=float(np.mean(aps)),
        num_doctors_evaluated=doctors_evaluated,
        details=details,
    )


def temporal_split(
    interactions: list[Interaction],
    train_ratio: float = 0.8,
) -> tuple[list[Interaction], list[Interaction]]:
    """
    Split temporal: ordena las interacciones por timestamp y separa en train/test.
    El split temporal es más realista que el aleatorio porque simula el escenario
    real donde entrenamos con el pasado y predecimos el futuro.
    """
    sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
    split_idx = int(len(sorted_interactions) * train_ratio)
    return sorted_interactions[:split_idx], sorted_interactions[split_idx:]
