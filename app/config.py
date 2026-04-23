from datetime import datetime, timezone

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    app_name: str = "Sinapsistencia ML — Sistema de Recomendación y Evaluación de Riesgo"
    app_version: str = "2.0.0"
    debug: bool = False

    # Parámetros del modelo Content-Based (TF-IDF + coseno)
    tfidf_max_features: int = 500
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2

    # Parámetros del modelo Random Forest (evaluación de riesgo)
    rf_n_estimators: int = 100
    rf_max_depth: int = 8
    rf_random_state: int = 42

    # Recomendaciones por defecto
    default_top_k: int = 10

    # Rutas de modelos serializados
    models_dir: str = "artifacts"

    # Supabase (para training pipeline)
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Modelo de riesgo
    risk_model_version: str = "1.0.0"


settings = Settings()


def get_model_version_tag() -> str:
    """Genera un tag de versión con timestamp para los artefactos."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"v{settings.app_version}_{ts}"
