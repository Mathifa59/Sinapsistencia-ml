"""Tests para los modelos ML del sistema de recomendación y riesgo."""

import pytest
from app.domain.entities import (
    DoctorProfile,
    LawyerProfile,
    Interaction,
    RecommendationRequest,
    RiskAssessmentRequest,
)
from app.models.content_based import ContentBasedRecommender
from app.models.collaborative import CollaborativeRecommender
from app.models.hybrid import HybridRecommender
from app.models.risk_assessment import RiskAssessmentModel
from datetime import datetime


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_lawyers() -> list[LawyerProfile]:
    return [
        LawyerProfile(
            id="law-001",
            name="Dr. García",
            specialties=["Responsabilidad Civil Médica", "Derecho Penal Médico"],
            medical_areas=["Cardiología", "Cirugía General"],
            years_experience=10,
            resolved_cases=50,
            rating=4.5,
        ),
        LawyerProfile(
            id="law-002",
            name="Dra. López",
            specialties=["Negligencia Médica"],
            medical_areas=["Neurología", "Oncología"],
            years_experience=8,
            resolved_cases=30,
            rating=4.2,
        ),
        LawyerProfile(
            id="law-003",
            name="Dr. Martínez",
            specialties=["Derecho Sanitario"],
            medical_areas=["Pediatría"],
            years_experience=15,
            resolved_cases=100,
            rating=4.8,
        ),
    ]


@pytest.fixture
def sample_doctor() -> DoctorProfile:
    return DoctorProfile(
        id="doc-001",
        name="Dr. Rodríguez",
        specialty="Cardiología",
        sub_specialties=["Cirugía Cardiovascular"],
        hospital="Hospital Nacional",
        years_experience=12,
    )


@pytest.fixture
def sample_interactions() -> list[Interaction]:
    return [
        Interaction(doctor_id="doc-001", lawyer_id="law-001", accepted=True, rating=5.0,
                    timestamp=datetime(2024, 1, 1)),
        Interaction(doctor_id="doc-001", lawyer_id="law-002", accepted=False,
                    timestamp=datetime(2024, 1, 15)),
        Interaction(doctor_id="doc-002", lawyer_id="law-001", accepted=True, rating=4.0,
                    timestamp=datetime(2024, 2, 1)),
        Interaction(doctor_id="doc-002", lawyer_id="law-003", accepted=True, rating=4.5,
                    timestamp=datetime(2024, 2, 15)),
        Interaction(doctor_id="doc-003", lawyer_id="law-002", accepted=True, rating=3.0,
                    timestamp=datetime(2024, 3, 1)),
        Interaction(doctor_id="doc-003", lawyer_id="law-001", accepted=True,
                    timestamp=datetime(2024, 3, 15)),
    ]


# ─── Content-Based Tests ────────────────────────────────────────────────────


class TestContentBased:
    def test_fit(self, sample_lawyers):
        model = ContentBasedRecommender()
        model.fit(sample_lawyers)
        assert model.is_fitted
        assert model.num_lawyers == 3

    def test_recommend(self, sample_lawyers, sample_doctor):
        model = ContentBasedRecommender()
        model.fit(sample_lawyers)
        recs = model.recommend(sample_doctor, top_k=3)
        assert len(recs) > 0
        assert all(0.0 <= r.score <= 1.0 for r in recs)
        assert recs[0].model_used == "content"

    def test_cardiologist_gets_cardiology_lawyer_first(self, sample_lawyers, sample_doctor):
        """Un cardiólogo debería recibir primero al abogado con área médica de cardiología."""
        model = ContentBasedRecommender()
        model.fit(sample_lawyers)
        recs = model.recommend(sample_doctor, top_k=3)
        # law-001 tiene "Cardiología" en medical_areas
        assert recs[0].lawyer_id == "law-001"

    def test_empty_lawyers_raises(self):
        model = ContentBasedRecommender()
        with pytest.raises(ValueError):
            model.fit([])


# ─── Collaborative Tests ────────────────────────────────────────────────────


class TestCollaborative:
    def test_fit(self, sample_interactions, sample_lawyers):
        model = CollaborativeRecommender(n_components=2)
        model.fit(sample_interactions, sample_lawyers)
        assert model.is_fitted
        assert model.num_doctors == 3

    def test_dynamic_components(self, sample_interactions, sample_lawyers):
        """SVD n_components se ajusta al tamaño real de la matriz."""
        model = CollaborativeRecommender(n_components=50)
        model.fit(sample_interactions, sample_lawyers)
        assert model.is_fitted
        # 3 doctors × 3 lawyers → max components = min(3,3)-1 = 2
        assert model._svd.n_components <= 2

    def test_recommend_known_doctor(self, sample_interactions, sample_lawyers):
        model = CollaborativeRecommender(n_components=2)
        model.fit(sample_interactions, sample_lawyers)
        recs = model.recommend("doc-001", top_k=3)
        assert len(recs) > 0

    def test_cold_start_empty(self, sample_interactions, sample_lawyers):
        model = CollaborativeRecommender(n_components=2)
        model.fit(sample_interactions, sample_lawyers)
        recs = model.recommend("unknown-doctor", top_k=3)
        assert recs == []

    def test_insufficient_interactions(self, sample_lawyers):
        model = CollaborativeRecommender()
        with pytest.raises(ValueError, match="al menos 5"):
            model.fit([
                Interaction(doctor_id="d1", lawyer_id="l1", accepted=True),
            ], sample_lawyers)


# ─── Hybrid Tests ────────────────────────────────────────────────────────────


class TestHybrid:
    def test_fit_without_interactions(self, sample_lawyers):
        model = HybridRecommender()
        model.fit(sample_lawyers, interactions=[])
        assert model.is_fitted
        assert not model.collaborative_model.is_fitted

    def test_fit_with_interactions(self, sample_lawyers, sample_interactions):
        model = HybridRecommender()
        model.fit(sample_lawyers, sample_interactions)
        assert model.is_fitted
        assert model.collaborative_model.is_fitted

    def test_recommend_includes_explainability(self, sample_lawyers, sample_doctor):
        model = HybridRecommender()
        model.fit(sample_lawyers)
        recs = model.recommend(sample_doctor, top_k=3)
        assert len(recs) > 0
        # Debe incluir feature_importance y reasons
        first = recs[0]
        assert isinstance(first.feature_importance, list)
        assert isinstance(first.reasons, list)
        assert len(first.reasons) > 0

    def test_model_info(self, sample_lawyers, sample_doctor):
        model = HybridRecommender()
        model.fit(sample_lawyers)
        info = model.get_model_info(sample_doctor.id)
        assert "model" in info
        assert "alpha_content" in info


# ─── Risk Assessment Tests ───────────────────────────────────────────────────


class TestRiskAssessment:
    def test_basic_assessment(self):
        model = RiskAssessmentModel()
        response = model.assess(RiskAssessmentRequest(
            specialty="Cardiología",
            priority="media",
        ))
        assert 0.0 <= response.risk_score <= 1.0
        assert response.risk_level in ("bajo", "moderado", "alto", "critico")
        assert len(response.risk_factors) == 7
        assert len(response.recommendations) > 0

    def test_high_risk_case(self):
        model = RiskAssessmentModel()
        response = model.assess(RiskAssessmentRequest(
            specialty="Neurocirugía",
            priority="critica",
            procedure_complexity="alta",
            documentation_complete=False,
            informed_consent=False,
            has_prior_complaints=True,
            time_since_incident_days=5,
        ))
        assert response.risk_score > 0.5
        assert response.risk_level in ("alto", "critico")
        # Debe tener recomendaciones urgentes
        urgent = [r for r in response.recommendations if "URGENTE" in r]
        assert len(urgent) >= 2

    def test_low_risk_case(self):
        model = RiskAssessmentModel()
        response = model.assess(RiskAssessmentRequest(
            specialty="Medicina Preventiva",
            priority="baja",
            procedure_complexity="baja",
            documentation_complete=True,
            informed_consent=True,
            has_prior_complaints=False,
            time_since_incident_days=400,
        ))
        assert response.risk_score < 0.3
        assert response.risk_level in ("bajo", "moderado")

    def test_factors_sum_to_weights(self):
        """Las contribuciones deben sumar al risk_score."""
        model = RiskAssessmentModel()
        response = model.assess(RiskAssessmentRequest(
            specialty="Cardiología",
            priority="alta",
        ))
        total = sum(f.contribution for f in response.risk_factors)
        assert abs(total - response.risk_score) < 0.01


# ─── Request Contract Tests ─────────────────────────────────────────────────


class TestRequestContract:
    def test_native_format(self):
        """Formato nativo con doctor: DoctorProfile."""
        req = RecommendationRequest(
            doctor=DoctorProfile(id="doc-1", specialty="Cardiología"),
            top_k=5,
        )
        doctor = req.resolve_doctor()
        assert doctor.id == "doc-1"
        assert doctor.specialty == "Cardiología"

    def test_frontend_format(self):
        """Formato frontend con doctor_id + doctor_profile."""
        req = RecommendationRequest(
            doctor_id="uuid-123",
            doctor_profile={
                "specialty": "Neurología",
                "hospital": "Hospital Nacional",
                "years_experience": 10,
            },
            top_k=10,
        )
        doctor = req.resolve_doctor()
        assert doctor.id == "uuid-123"
        assert doctor.specialty == "Neurología"
        assert doctor.hospital == "Hospital Nacional"

    def test_no_doctor_raises(self):
        """Sin doctor ni doctor_id debe fallar."""
        req = RecommendationRequest(top_k=5)
        with pytest.raises(ValueError, match="Se requiere"):
            req.resolve_doctor()
