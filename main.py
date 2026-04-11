"""
Sinapsistencia ML — Sistema de Recomendación Médico-Legal
=========================================================
Microservicio FastAPI que expone el motor de recomendación basado en ML
para el matching entre médicos y abogados especializados.

Arquitectura del sistema:
    1. Content-Based Filtering  → similitud de perfiles (TF-IDF + coseno)
    2. Collaborative Filtering  → patrones de matches históricos (SVD)
    3. Hybrid Recommender       → combinación ponderada de ambos modelos
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
    # Cleanup al cerrar (si fuera necesario)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Motor de Recomendación Médico-Legal

    Utiliza Machine Learning para recomendar abogados a médicos basándose en:

    - **Similitud de especialidades** (Content-Based Filtering con TF-IDF)
    - **Patrones históricos de matches exitosos** (Collaborative Filtering con SVD)
    - **Modelo híbrido** que combina ambos enfoques ponderadamente

    ### Algoritmos implementados
    - TF-IDF Vectorizer para representación de especialidades médico-legales
    - Similitud del coseno para medir afinidad entre perfiles
    - Descomposición en Valores Singulares (SVD) para factores latentes
    - Ponderación adaptativa según disponibilidad de datos históricos

    ### Métricas de evaluación
    - Precision@K, Recall@K, NDCG@K
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
app.include_router(recommendations.router, prefix="/api/v1", tags=["Recomendaciones"])
