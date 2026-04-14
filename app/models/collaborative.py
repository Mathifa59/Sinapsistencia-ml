"""
Modelo 2: Collaborative Filtering con SVD (Matrix Factorization)
================================================================
Aprende de los patrones históricos de matches entre médicos y abogados.

Algoritmo — Descomposición en Valores Singulares (SVD):
    1. Se construye la matriz de interacciones R de tamaño (n_doctores × n_abogados)
          R[i][j] = score de la interacción entre doctor i y abogado j
                    (1.0 si aceptó, 0.0 si rechazó, NaN si no hubo interacción)

    2. Se aplica Truncated SVD para factorizar R en tres matrices:
          R ≈ U × Σ × Vᵀ
          donde:
            U  (n_doctores × k)   : factores latentes de los médicos
            Σ  (k × k)            : valores singulares (importancia de cada factor)
            Vᵀ (k × n_abogados)  : factores latentes de los abogados
          k = n_components (hiperparámetro, por defecto 50)

    3. Para predecir el score de un par (doctor_i, abogado_j):
          score(i, j) = U[i] · Σ · V[j]ᵀ
          (dot product de los vectores de factores latentes)

    4. Para un nuevo doctor (sin historial), no se puede aplicar collaborative.
       → El sistema cae back al modelo Content-Based.

¿Qué capturan los factores latentes?
    Patrones implícitos no definidos explícitamente: por ejemplo, que los
    médicos del Hospital Nacional prefieren abogados con perfil más conservador,
    o que doctores con muchos años de experiencia tienden a elegir abogados
    especializados en casos complejos. El modelo los aprende solo de los datos.

Ventajas sobre Content-Based:
    - Captura preferencias que no están en el texto del perfil
    - Mejora con el tiempo a medida que se acumulan interacciones
    - Puede recomendar abogados que el médico no conocería por similitud de texto

Limitaciones:
    - Cold-start: no funciona para médicos sin historial (se usa Content-Based)
    - Requiere suficientes interacciones para ser significativo (~50+)
"""

import numpy as np
import joblib
import os
from typing import Sequence

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

from app.domain.entities import Interaction, RecommendationScore, LawyerProfile
from app.config import settings


class CollaborativeRecommender:
    """
    Recommender colaborativo basado en SVD (Matrix Factorization).

    Atributos internos:
        _svd             : TruncatedSVD de scikit-learn
        _doctor_factors  : U × Σ — factores latentes de médicos (n_doctores × k)
        _lawyer_factors  : Vᵀ — factores latentes de abogados (k × n_abogados)
        _doctor_ids      : Mapeo índice → doctor_id
        _lawyer_ids      : Mapeo índice → lawyer_id
        _doctor_index    : Mapeo doctor_id → índice en _doctor_factors
        _is_fitted       : Flag que indica si el modelo fue entrenado
    """

    def __init__(self, n_components: int = settings.svd_n_components):
        self._svd = TruncatedSVD(
            n_components=n_components,
            random_state=settings.svd_random_state,
            algorithm="randomized",  # más eficiente para matrices grandes
        )
        self._score_scaler = MinMaxScaler()
        self._doctor_factors: np.ndarray | None = None   # U × Σ
        self._lawyer_factors: np.ndarray | None = None   # Vᵀ
        self._doctor_ids: list[str] = []
        self._lawyer_ids: list[str] = []
        self._doctor_index: dict[str, int] = {}
        self._lawyer_index: dict[str, int] = {}
        self._lawyers: dict[str, LawyerProfile] = {}
        self._is_fitted: bool = False
        self._n_components = n_components

    # ─── Entrenamiento ────────────────────────────────────────────────────────

    def fit(
        self,
        interactions: Sequence[Interaction],
        lawyers: Sequence[LawyerProfile],
    ) -> "CollaborativeRecommender":
        """
        Entrena el modelo construyendo y factorizando la matriz de interacciones.

        Pasos:
            1. Extraer todos los IDs únicos de doctores y abogados
            2. Construir la matriz R (doctores × abogados) con los scores
            3. Rellenar los valores faltantes (NaN → 0 para SVD)
            4. Aplicar TruncatedSVD para factorizar R ≈ U × Σ × Vᵀ
            5. Guardar los factores latentes

        Args:
            interactions: Historial de interacciones doctor-abogado.
            lawyers     : Perfiles de abogados (para metadatos en la respuesta).

        Returns:
            self (para encadenamiento)
        """
        if len(interactions) < 5:
            raise ValueError(
                f"Se necesitan al menos 5 interacciones para entrenar el modelo "
                f"colaborativo. Hay {len(interactions)}."
            )

        # Almacena perfiles para lookup
        self._lawyers = {l.id: l for l in lawyers}

        # ── Paso 1: Identificar todos los actores ─────────────────────────────
        doctor_ids_set: set[str] = set()
        lawyer_ids_set: set[str] = set()
        for interaction in interactions:
            doctor_ids_set.add(interaction.doctor_id)
            lawyer_ids_set.add(interaction.lawyer_id)

        self._doctor_ids = sorted(doctor_ids_set)
        self._lawyer_ids = sorted(lawyer_ids_set)
        self._doctor_index = {d: i for i, d in enumerate(self._doctor_ids)}
        self._lawyer_index = {l: i for i, l in enumerate(self._lawyer_ids)}

        n_doctors = len(self._doctor_ids)
        n_lawyers = len(self._lawyer_ids)

        # ── Paso 2: Construir matriz de ratings ───────────────────────────────
        # Inicializa con 0 (interacción desconocida → neutral)
        R = np.zeros((n_doctors, n_lawyers), dtype=np.float32)

        for interaction in interactions:
            d_idx = self._doctor_index[interaction.doctor_id]
            l_idx = self._lawyer_index[interaction.lawyer_id]
            # Usa implicit_score: 1.0 aceptado, 0.0 rechazado (o rating normalizado)
            R[d_idx, l_idx] = interaction.implicit_score

        # ── Paso 3: Aplicar Truncated SVD ─────────────────────────────────────
        # Ajusta n_components dinámicamente al tamaño real de la matriz.
        # SVD requiere n_components < min(n_rows, n_cols).
        # Para datasets pequeños (ej: 10 doctores × 12 abogados), usar
        # n_components=50 es absurdo — se reduce a max(1, min(dims)-1).
        max_possible = min(n_doctors, n_lawyers) - 1
        actual_components = max(1, min(self._n_components, max_possible))
        if actual_components != self._n_components:
            self._svd = TruncatedSVD(
                n_components=actual_components,
                random_state=settings.svd_random_state,
            )

        # SVD: R ≈ U × Σ × Vᵀ
        # fit_transform retorna U × Σ directamente
        self._doctor_factors = self._svd.fit_transform(R)   # (n_doctors × k)
        self._lawyer_factors = self._svd.components_         # (k × n_lawyers)

        # ── Paso 4: Pre-calcular todos los scores para scaling ─────────────────
        all_scores = self._doctor_factors @ self._lawyer_factors  # (n_doctors × n_lawyers)
        self._score_scaler.fit(all_scores.reshape(-1, 1))

        self._is_fitted = True
        return self

    # ─── Predicción ───────────────────────────────────────────────────────────

    def predict_score(self, doctor_id: str, lawyer_id: str) -> float:
        """
        Predice el score de afinidad entre un doctor y un abogado.

        Cálculo: U[doctor] · Vᵀ[lawyer] (producto punto de factores latentes)
        El resultado se normaliza a [0, 1].

        Returns 0.0 si el doctor o el abogado no están en el historial.
        """
        self._assert_fitted()
        if doctor_id not in self._doctor_index or lawyer_id not in self._lawyer_index:
            return 0.0

        d_idx = self._doctor_index[doctor_id]
        l_idx = self._lawyer_index[lawyer_id]

        # Producto punto entre factores latentes del doctor y del abogado
        raw_score = float(self._doctor_factors[d_idx] @ self._lawyer_factors[:, l_idx])

        # Normaliza a [0, 1]
        normalized = self._score_scaler.transform([[raw_score]])[0][0]
        return float(np.clip(normalized, 0.0, 1.0))

    def recommend(
        self,
        doctor_id: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[RecommendationScore]:
        """
        Genera recomendaciones para un médico con historial previo.

        Args:
            doctor_id : ID del médico (debe estar en el historial).
            top_k     : Número máximo de recomendaciones.
            min_score : Score mínimo para incluir una recomendación.

        Returns:
            Lista de RecommendationScore o lista vacía si el doctor
            no tiene historial (cold-start → usar Content-Based).
        """
        self._assert_fitted()

        if doctor_id not in self._doctor_index:
            return []  # cold-start: sin historial → delegar a content-based

        d_idx = self._doctor_index[doctor_id]

        # Scores para todos los abogados conocidos
        raw_scores = self._doctor_factors[d_idx] @ self._lawyer_factors  # (n_lawyers,)

        # Normaliza todos los scores de este doctor a [0, 1]
        normalized_scores = self._score_scaler.transform(
            raw_scores.reshape(-1, 1)
        ).flatten()
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

        # Ordena de mayor a menor
        ranked_indices = np.argsort(normalized_scores)[::-1]

        results: list[RecommendationScore] = []
        for idx in ranked_indices:
            score = float(normalized_scores[idx])
            if score < min_score:
                break
            if len(results) >= top_k:
                break

            lawyer_id = self._lawyer_ids[idx]
            lawyer = self._lawyers.get(lawyer_id)
            lawyer_name = lawyer.name if lawyer else lawyer_id

            results.append(
                RecommendationScore(
                    lawyer_id=lawyer_id,
                    lawyer_name=lawyer_name,
                    score=round(score, 4),
                    content_score=0.0,
                    collaborative_score=round(score, 4),
                    matched_specialties=[],
                    model_used="collaborative",
                )
            )

        return results

    def has_doctor(self, doctor_id: str) -> bool:
        """Indica si el médico tiene historial en el modelo."""
        return doctor_id in self._doctor_index

    # ─── Serialización ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serializa el modelo entrenado a disco."""
        self._assert_fitted()
        os.makedirs(path, exist_ok=True)
        joblib.dump(self._svd, os.path.join(path, "svd.pkl"))
        joblib.dump(self._score_scaler, os.path.join(path, "scaler_collab.pkl"))
        joblib.dump(self._doctor_factors, os.path.join(path, "doctor_factors.pkl"))
        joblib.dump(self._lawyer_factors, os.path.join(path, "lawyer_factors.pkl"))
        joblib.dump(self._doctor_ids, os.path.join(path, "doctor_ids.pkl"))
        joblib.dump(self._lawyer_ids, os.path.join(path, "lawyer_ids_collab.pkl"))
        joblib.dump(self._doctor_index, os.path.join(path, "doctor_index.pkl"))
        joblib.dump(self._lawyer_index, os.path.join(path, "lawyer_index.pkl"))
        joblib.dump(self._lawyers, os.path.join(path, "lawyers_collab.pkl"))

    def load(self, path: str) -> "CollaborativeRecommender":
        """Carga un modelo previamente serializado desde disco."""
        self._svd = joblib.load(os.path.join(path, "svd.pkl"))
        self._score_scaler = joblib.load(os.path.join(path, "scaler_collab.pkl"))
        self._doctor_factors = joblib.load(os.path.join(path, "doctor_factors.pkl"))
        self._lawyer_factors = joblib.load(os.path.join(path, "lawyer_factors.pkl"))
        self._doctor_ids = joblib.load(os.path.join(path, "doctor_ids.pkl"))
        self._lawyer_ids = joblib.load(os.path.join(path, "lawyer_ids_collab.pkl"))
        self._doctor_index = joblib.load(os.path.join(path, "doctor_index.pkl"))
        self._lawyer_index = joblib.load(os.path.join(path, "lawyer_index.pkl"))
        self._lawyers = joblib.load(os.path.join(path, "lawyers_collab.pkl"))
        self._is_fitted = True
        return self

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llama a .fit(interactions, lawyers) primero."
            )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def num_doctors(self) -> int:
        return len(self._doctor_ids)

    @property
    def num_lawyers(self) -> int:
        return len(self._lawyer_ids)

    @property
    def explained_variance_ratio(self) -> float:
        """Varianza explicada por los componentes SVD (métrica de calidad)."""
        if not self._is_fitted:
            return 0.0
        return float(np.sum(self._svd.explained_variance_ratio_))
