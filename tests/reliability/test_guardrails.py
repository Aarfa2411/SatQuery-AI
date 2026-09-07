"""
Unit & Reliability Tests for NumericalGuardrail (Workstream H)
Validates:
1. Unit-type first matching.
2. Order-of-mention resolution.
3. Conversational paraphrase tolerance (+/- 5%).
4. Numerical drift interception & substitution.
5. Coordinate point-in-polygon verification.
"""

import pytest
from satquery.evaluation.guardrails import NumericalGuardrail


@pytest.fixture
def guardrail():
    return NumericalGuardrail(tolerance=0.05)


def test_paraphrased_number_within_tolerance_passes(guardrail):
    """'around 12 hectares' for 11.93 ha has 0.58% error -> should pass."""
    text = "Satellite differencing revealed around 12 hectares of forest loss."
    det_nums = {"area_ha": 11.93}

    passed, violations, rectified = guardrail.verify(text, det_nums)
    assert passed is True
    assert len(violations) == 0
    assert "around 12 hectares" in rectified


def test_numerical_drift_exceeding_tolerance_is_flagged_and_rectified(guardrail):
    """'25 hectares' for 11.93 ha has >100% error -> should flag and substitute."""
    text = "Analysis shows 25 hectares of deforestation in the reserve."
    det_nums = {"area_ha": 11.93}

    passed, violations, rectified = guardrail.verify(text, det_nums)
    assert passed is False
    assert len(violations) == 1
    assert violations[0].violation_type == "numerical_drift"
    assert "11.93 hectares" in rectified


def test_unit_type_isolation_prevents_wrong_key_matching(guardrail):
    """Area units must not be matched against percentage keys."""
    text = "Water body expanded by 15.0% over an affected area of 45.0 ha."
    det_nums = {
        "change_pct": 14.8,  # Within 5% of 15.0%
        "area_ha": 44.5,     # Within 5% of 45.0 ha
    }

    passed, violations, rectified = guardrail.verify(text, det_nums)
    assert passed is True
    assert len(violations) == 0


def test_order_of_mention_for_multiple_values_of_same_unit(guardrail):
    """When before and after areas appear, they map sequentially to area keys."""
    text = "Baseline forest was 100.0 ha and shrunk to 80.0 ha."
    det_nums = {
        "baseline_area_ha": 98.5,  # Within 5% of 100.0
        "current_area_ha": 81.2,   # Within 5% of 80.0
    }

    passed, violations, rectified = guardrail.verify(text, det_nums)
    assert passed is True
    assert len(violations) == 0


def test_ungrounded_number_claim_is_flagged(guardrail):
    """Number with no corresponding tool metric must be flagged."""
    text = "There are 500 meters of road blocked."
    det_nums = {"area_ha": 10.0}  # No distance_m metric

    passed, violations, _ = guardrail.verify(text, det_nums)
    assert passed is False
    assert any(v.violation_type == "ungrounded_claim" for v in violations)


def test_coordinate_point_in_polygon_validation(guardrail):
    """Validates whether claimed coordinates fall inside the spatial evidence polygon."""
    sample_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [76.0, 12.0],
            [77.0, 12.0],
            [77.0, 13.0],
            [76.0, 13.0],
            [76.0, 12.0],
        ]]
    }

    # Inside polygon: lat 12.5, lon 76.5
    valid_text = "Detection confirmed at lat: 12.5, lon: 76.5 covering 10.0 ha."
    passed_valid, violations_valid, _ = guardrail.verify(
        valid_text, {"area_ha": 10.0}, geojson_geometry=sample_geojson
    )
    assert passed_valid is True

    # Outside polygon: lat 15.0, lon 79.0 (spatial drift hallucination)
    hallucinated_text = "Detection confirmed at lat: 15.0, lon: 79.0 covering 10.0 ha."
    passed_drift, violations_drift, _ = guardrail.verify(
        hallucinated_text, {"area_ha": 10.0}, geojson_geometry=sample_geojson
    )
    assert passed_drift is False
    assert any(v.violation_type == "spatial_drift" for v in violations_drift)
