"""
Modelo 1: Content-Based Filtering
==================================
Recomienda abogados a médicos basándose en la similitud entre sus perfiles.

Algoritmo:
    1. Cada abogado se representa como un vector de features:
          v_abogado = [vector_tfidf | features_numericas_normalizadas]

    2. La consulta del médico se transforma al mismo espacio vectorial:
          v_doctor = [vector_tfidf | features_numericas_normalizadas]

    3. La similitud entre doctor y cada abogado se calcula con coseno:
          similitud(d, a) = (v_doctor · v_abogado) / (‖v_doctor‖ · ‖v_abogado‖)

    4. Se retornan los top-K abogados con mayor similitud.

¿Por qué similitud del coseno?
    Mide el ángulo entre vectores, no la magnitud. Eso hace que un perfil
    con muchas palabras no tenga ventaja injusta sobre uno conciso.
    Valor en [0, 1]: 1 = perfiles idénticos, 0 = sin relación alguna.

Ventajas:
    - No requiere historial de interacciones (funciona desde el primer día)
    - Explicable: se puede mostrar qué especialidades coincidieron
    - Actualizable: re-entrenamiento rápido al agregar nuevos abogados

Limitaciones:
    - No captura preferencias implícitas del médico
    - Dos abogados con el mismo texto pero diferente rendimiento real
      tendrán el mismo score → se complementa con Collaborative Filtering
"""

import numpy as np
import joblib
import os
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from app.domain.entities import DoctorProfile, LawyerProfile, RecommendationScore
from app.data.preprocessing import (
    build_lawyer_text,
    build_doctor_text,
    extract_lawyer_numerical_features,
    extract_doctor_numerical_features,
    find_matching_specialties,
)
from app.config import settings


class ContentBasedRecommender:
    """
    Recommender basado en contenido usando TF-IDF + similitud del coseno.

    Atributos internos:
        _tfidf         : TfidfVectorizer entrenado sobre los perfiles de abogados
        _scaler        : MinMaxScaler entrenado sobre features numéricas
        _lawyer_matrix : Matriz (n_abogados × n_features) — una fila por abogado
        _lawyer_ids    : Lista de IDs en el mismo orden que las filas de _lawyer_matrix
        _lawyers       : Perfiles completos para extraer metadatos en la respuesta
        _is_fitted     : Flag que indica si el modelo fue entrenado
    """

    def __init__(self):
        self._tfidf = TfidfVectorizer(
            max_features=settings.tfidf_max_features,
            ngram_range=(settings.tfidf_ngram_min, settings.tfidf_ngram_max),
            analyzer="word",
            sublinear_tf=True,    # aplica log(1+tf) → reduce el efecto de términos muy frecuentes
            min_df=1,             # incluye términos que aparecen en al menos 1 documento
        )
        self._scaler = MinMaxScaler()
        self._lawyer_matrix: csr_matrix | None = None
        self._lawyer_ids: list[str] = []
        self._lawyers: dict[str, LawyerProfile] = {}
        self._is_fitted: bool = False

    # ─── Entrenamiento ────────────────────────────────────────────────────────

    def fit(self, lawyers: Sequence[LawyerProfile]) -> "ContentBasedRecommender":
        """
        Entrena el modelo construyendo la matriz de features de todos los abogados.

        Pasos:
            1. Construir el texto de cada abogado
            2. Ajustar y transformar con TF-IDF → matriz dispersa (sparse)
            3. Extraer y normalizar features numéricas → matriz densa
            4. Concatenar ambas matrices horizontalmente → _lawyer_matrix

        Args:
            lawyers: Lista de perfiles de abogados a indexar.

        Returns:
            self (para encadenamiento)
        """
        if not lawyers:
            raise ValueError("Se necesita al menos un abogado para entrenar el modelo.")

        lawyer_list = list(lawyers)

        # Almacena perfiles para lookup posterior
        self._lawyers = {l.id: l for l in lawyer_list}
        self._lawyer_ids = [l.id for l in lawyer_list]

        # ── Paso 1: Vectorización TF-IDF ──────────────────────────────────────
        lawyer_texts = [build_lawyer_text(l) for l in lawyer_list]
        tfidf_matrix = self._tfidf.fit_transform(lawyer_texts)
        # tfidf_matrix: (n_abogados × max_features) — dispersa

        # ── Paso 2: Features numéricas normalizadas ────────────────────────────
        numerical_features = np.array([
            extract_lawyer_numerical_features(l) for l in lawyer_list
        ])
        # Ajusta el scaler y transforma: cada columna queda en [0, 1]
        numerical_normalized = self._scaler.fit_transform(numerical_features)
        numerical_sparse = csr_matrix(numerical_normalized)

        # ── Paso 3: Concatenación horizontal ──────────────────────────────────
        # Vector final: [tfidf_features | num_features]
        self._lawyer_matrix = hstack([tfidf_matrix, numerical_sparse])

        self._is_fitted = True
        return self

    # ─── Predicción ───────────────────────────────────────────────────────────

    def recommend(
        self,
        doctor: DoctorProfile,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[RecommendationScore]:
        """
        Genera las top-K recomendaciones de abogados para un médico.

        Args:
            doctor   : Perfil del médico que solicita recomendaciones.
            top_k    : Número máximo de recomendaciones a retornar.
            min_score: Score mínimo para incluir una recomendación (filtro).

        Returns:
            Lista de RecommendationScore ordenada de mayor a menor similitud.
        """
        self._assert_fitted()

        # ── Construir vector del médico ────────────────────────────────────────
        doctor_text = build_doctor_text(doctor)

        # Transforma el texto del médico usando el vocabulario ya aprendido
        # (transform, no fit_transform → usa el vocabulario del entrenamiento)
        doctor_tfidf = self._tfidf.transform([doctor_text])

        # Features numéricas del médico (normalizadas con el mismo scaler)
        # El scaler fue entrenado con 3 features de abogados; para el doctor
        # usamos solo años de experiencia → padding con ceros para compatibilidad
        doc_numerical_raw = extract_doctor_numerical_features(doctor)
        doc_numerical_padded = np.zeros((1, 3))
        doc_numerical_padded[0, 0] = doc_numerical_raw[0]  # años experiencia
        doc_numerical_norm = self._scaler.transform(doc_numerical_padded)
        doc_numerical_sparse = csr_matrix(doc_numerical_norm)

        doctor_vector = hstack([doctor_tfidf, doc_numerical_sparse])

        # ── Similitud del coseno contra todos los abogados ────────────────────
        # cosine_similarity retorna array (1 × n_abogados)
        similarities = cosine_similarity(doctor_vector, self._lawyer_matrix)[0]

        # ── Ordenar y filtrar ─────────────────────────────────────────────────
        ranked_indices = np.argsort(similarities)[::-1]  # mayor a menor

        results: list[RecommendationScore] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            if len(results) >= top_k:
                break

            lawyer_id = self._lawyer_ids[idx]
            lawyer = self._lawyers[lawyer_id]

            results.append(
                RecommendationScore(
                    lawyer_id=lawyer_id,
                    lawyer_name=lawyer.name,
                    score=round(score, 4),
                    content_score=round(score, 4),
                    collaborative_score=0.0,
                    matched_specialties=find_matching_specialties(doctor, lawyer),
                    model_used="content",
                )
            )

        return results

    def get_similarity_score(self, doctor: DoctorProfile, lawyer_id: str) -> float:
        """Retorna el score de similitud entre un doctor y un abogado específico."""
        self._assert_fitted()
        if lawyer_id not in self._lawyers:
            return 0.0

        lawyer_idx = self._lawyer_ids.index(lawyer_id)
        doctor_text = build_doctor_text(doctor)
        doctor_tfidf = self._tfidf.transform([doctor_text])
        doc_numerical_padded = np.zeros((1, 3))
        doc_numerical_padded[0, 0] = float(doctor.years_experience)
        doc_numerical_norm = self._scaler.transform(doc_numerical_padded)
        doctor_vector = hstack([doctor_tfidf, csr_matrix(doc_numerical_norm)])

        lawyer_vector = self._lawyer_matrix[lawyer_idx]
        return float(cosine_similarity(doctor_vector, lawyer_vector)[0][0])

    # ─── Serialización ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serializa el modelo entrenado a disco usando joblib."""
        self._assert_fitted()
        os.makedirs(path, exist_ok=True)
        joblib.dump(self._tfidf, os.path.join(path, "tfidf.pkl"))
        joblib.dump(self._scaler, os.path.join(path, "scaler_content.pkl"))
        joblib.dump(self._lawyer_matrix, os.path.join(path, "lawyer_matrix.pkl"))
        joblib.dump(self._lawyer_ids, os.path.join(path, "lawyer_ids.pkl"))
        joblib.dump(self._lawyers, os.path.join(path, "lawyers.pkl"))

    def load(self, path: str) -> "ContentBasedRecommender":
        """Carga un modelo previamente serializado desde disco."""
        self._tfidf = joblib.load(os.path.join(path, "tfidf.pkl"))
        self._scaler = joblib.load(os.path.join(path, "scaler_content.pkl"))
        self._lawyer_matrix = joblib.load(os.path.join(path, "lawyer_matrix.pkl"))
        self._lawyer_ids = joblib.load(os.path.join(path, "lawyer_ids.pkl"))
        self._lawyers = joblib.load(os.path.join(path, "lawyers.pkl"))
        self._is_fitted = True
        return self

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llama a .fit(lawyers) primero."
            )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def vocabulary_size(self) -> int:
        """Tamaño del vocabulario TF-IDF aprendido."""
        self._assert_fitted()
        return len(self._tfidf.vocabulary_)

    @property
    def num_lawyers(self) -> int:
        """Número de abogados indexados."""
        return len(self._lawyer_ids)
