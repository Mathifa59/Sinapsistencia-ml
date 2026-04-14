"""
Sinapsistencia ML — Sistema de Recomendación y Evaluación de Riesgo Médico-Legal
=================================================================================
Microservicio FastAPI que expone:

1. Motor de Recomendación basado en ML:
    - Content-Based Filtering  → similitud de perfiles (TF-IDF + coseno)
    - Collaborative Filtering  → patrones de matches históricos (SVD)
    - Hybrid Recommender       → combinación ponderada adaptativa
    - Explicabilidad           → feature importance + razones textuales

2. Evaluación de Riesgo Médico-Legal:
    - Modelo basado en factores ponderados
    - Desglose de factores de riesgo
    - Recomendaciones de mitigación

3. Pipeline de Entrenamiento:
    - Desde datos de muestra (desarrollo)
    - Desde Supabase (producción con UUIDs reales)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import recommendations, health
from app.config import settings
from app.services.recommender import recommender_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos al iniciar el servidor."""
    await recommender_service.initialize()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Motor de Recomendación y Evaluación de Riesgo Médico-Legal

    ### Recomendaciones
    Utiliza Machine Learning para recomendar abogados a médicos basándose en:
    - **Similitud de especialidades** (Content-Based Filtering con TF-IDF)
    - **Patrones históricos de matches exitosos** (Collaborative Filtering con SVD)
    - **Modelo híbrido** que combina ambos enfoques ponderadamente
    - **Explicabilidad** con feature importance y razones textuales

    ### Evaluación de Riesgo (Épica 13)
    Analiza factores de riesgo médico-legal de un caso:
    - Riesgo por especialidad médica
    - Complejidad del procedimiento
    - Estado de documentación y consentimiento informado
    - Historial de quejas y factor temporal

    ### Entrenamiento
    - Datos de muestra para desarrollo
    - Pipeline Supabase para producción (UUIDs reales)

    ### Métricas de evaluación
    - Precision@K, Recall@K, NDCG@K, MAP
    """,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(recommendations.router, prefix="/api/v1", tags=["Recomendaciones y Riesgo"])
