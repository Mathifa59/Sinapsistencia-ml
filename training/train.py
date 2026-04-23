"""
Pipeline de entrenamiento
=========================
Script principal para entrenar, evaluar y guardar los modelos ML.

Uso:
    python -m training.train

Pasos del pipeline:
    1. Cargar datos (abogados + historial de interacciones)
    2. Split temporal (80% train, 20% test)
    3. Entrenar HybridRecommender con datos de train
    4. Evaluar con datos de test (Precision@K, Recall@K, NDCG@K)
    5. Guardar modelos entrenados en artifacts/
    6. Imprimir reporte de métricas
"""

import json
import sys
from pathlib import Path

# Asegura que el directorio raíz esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain.entities import LawyerProfile, DoctorProfile, Interaction
from app.models.hybrid import HybridRecommender
from app.config import settings
from training.evaluate import evaluate_model, temporal_split


def load_data() -> tuple[list[LawyerProfile], list[DoctorProfile], list[Interaction]]:
    """Carga los datos desde los archivos JSON de muestra."""
    base = Path("data/sample")

    with open(base / "lawyers.json", encoding="utf-8") as f:
        lawyers = [LawyerProfile(**item) for item in json.load(f)]

    with open(base / "doctors.json", encoding="utf-8") as f:
        doctors = [DoctorProfile(**item) for item in json.load(f)]

    interactions: list[Interaction] = []
    interactions_path = base / "interactions.json"
    if interactions_path.exists():
        with open(interactions_path, encoding="utf-8") as f:
            interactions = [Interaction(**item) for item in json.load(f)]

    return lawyers, doctors, interactions


def run_training_pipeline() -> None:
    """Ejecuta el pipeline completo de entrenamiento y evaluación."""

    print("\n" + "=" * 55)
    print("  SINAPSISTENCIA ML — Pipeline de Entrenamiento")
    print("=" * 55)

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print("\n[1/5] Cargando datos...")
    lawyers, doctors, interactions = load_data()
    print(f"      Abogados  : {len(lawyers)}")
    print(f"      Médicos   : {len(doctors)}")
    print(f"      Interacciones: {len(interactions)}")

    # ── 2. Split temporal ─────────────────────────────────────────────────────
    print("\n[2/5] Split temporal (80/20)...")
    train_interactions, test_interactions = temporal_split(interactions, train_ratio=0.8)
    print(f"      Train: {len(train_interactions)} interacciones")
    print(f"      Test : {len(test_interactions)} interacciones")

    # ── 3. Entrenar ───────────────────────────────────────────────────────────
    print("\n[3/5] Entrenando HybridRecommender...")
    model = HybridRecommender()
    model.fit(lawyers=lawyers, interactions=train_interactions)

    print(f"      Content-Based     : entrenado")
    print(f"      Vocabulario TF-IDF: {model.content_model.vocabulary_size} términos")

    # ── 4. Evaluar ────────────────────────────────────────────────────────────
    print("\n[4/5] Evaluando con datos de test...")

    for k in [5, 10]:
        result = evaluate_model(
            model=model,
            test_doctors=doctors,
            test_interactions=test_interactions,
            k=k,
            model_name="hybrid",
        )
        print(result.summary())

    # ── 5. Guardar modelos ────────────────────────────────────────────────────
    print(f"[5/5] Guardando modelos en '{settings.models_dir}/'...")
    model.save(settings.models_dir)
    print("      Modelos guardados exitosamente.")
    print("\n  El servidor está listo para servir recomendaciones.\n")


if __name__ == "__main__":
    run_training_pipeline()
