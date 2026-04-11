from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Sinapsistencia ML — Sistema de Recomendación"
    app_version: str = "1.0.0"
    debug: bool = False

    # Parámetros del modelo Content-Based
    tfidf_max_features: int = 500
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2

    # Parámetros del modelo Collaborative Filtering
    svd_n_components: int = 50
    svd_random_state: int = 42

    # Parámetros del modelo Híbrido
    # α pondera content-based, β pondera collaborative
    # Se ajustan según la cantidad de historial disponible
    hybrid_alpha: float = 0.6    # peso content-based
    hybrid_beta: float = 0.4     # peso collaborative

    # Recomendaciones por defecto
    default_top_k: int = 10

    # Rutas de modelos serializados
    models_dir: str = "artifacts"

    class Config:
        env_file = ".env"


settings = Settings()
