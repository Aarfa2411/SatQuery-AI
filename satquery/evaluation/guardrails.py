"""
Numerical & Geospatial Anti-Hallucination Guardrail (P0 Deliverable)
Prevents LLM hallucinations by verifying generated numbers and coordinates against
deterministic raster/vector metrics using unit-type isolation, order-of-mention mapping,
and a +/-5% tolerance bound.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from shapely.geometry import Point, shape


# Regex for numerical quantities with units and optional conversational prefixes
NUMBER_UNIT_REGEX = re.compile(
    r'(?i)(?:around|about|approximately|nearly|exactly|roughly)?\s*'
    r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*'
    r'(hectares?|ha|sq\s*m|m\^2|m²|sq\s*km|km\^2|km²|percent|%|meters?|m\b|km\b)',
    re.IGNORECASE
)

# Regex for latitude / longitude coordinates
COORD_REGEX = re.compile(
    r'(?i)(?:lat(?:itude)?\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)[,\s]+lon(?:gitude)?\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)|'
    r'([+-]?\d+\.\d+)[°\s]*([NS])[,\s]+([+-]?\d+\.\d+)[°\s]*([EW]))',
    re.IGNORECASE
)

UNIT_TYPE_MAPPING = {
    "ha": "area",
    "hectare": "area",
    "hectares": "area",
    "sq m": "area_m2",
    "m^2": "area_m2",
    "m²": "area_m2",
    "sq km": "area_km2",
    "km^2": "area_km2",
    "km²": "area_km2",
    "%": "percentage",
    "percent": "percentage",
    "m": "distance_m",
    "meters": "distance_m",
    "meter": "distance_m",
    "km": "distance_km",
    "kilometers": "distance_km",
    "kilometer": "distance_km",
}


class GuardrailViolation:
    def __init__(self, violation_type: str, raw_claim: str, expected_value: Any, actual_value: Any, message: str):
        self.violation_type = violation_type  # "numerical_drift", "ungrounded_claim", "spatial_drift"
        self.raw_claim = raw_claim
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "raw_claim": self.raw_claim,
            "expected": self.expected_value,
            "actual": self.actual_value,
            "message": self.message,
        }


class NumericalGuardrail:
    """
    Validates text output against deterministic metrics and geometry.
    Enforces the core rule: LLMs must never invent or drift unverified numbers.
    """

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance  # 5% relative tolerance

    def extract_numbers_with_units(self, text: str) -> List[Tuple[float, str, str]]:
        """
        Extracts list of (numeric_val, unit_type, raw_string) from text.
        """
        results = []
        for match in NUMBER_UNIT_REGEX.finditer(text):
            val_str = match.group(1).replace(",", "")
            val = float(val_str)
            raw_unit = match.group(2).lower()
            unit_type = UNIT_TYPE_MAPPING.get(raw_unit, "unknown")
            results.append((val, unit_type, match.group(0).strip()))
        return results

    def extract_coordinates(self, text: str) -> List[Tuple[float, float, str]]:
        """
        Extracts (lat, lon, raw_string) from text.
        """
        results = []
        for match in COORD_REGEX.finditer(text):
            if match.group(1) is not None:
                lat = float(match.group(1))
                lon = float(match.group(2))
            else:
                lat = float(match.group(3)) * (-1 if match.group(4).upper() == "S" else 1)
                lon = float(match.group(5)) * (-1 if match.group(6).upper() == "W" else 1)
            results.append((lat, lon, match.group(0).strip()))
        return results

    def canonicalize_to_key(self, unit_type: str, val: float) -> Tuple[str, float]:
        """
        Maps unit type and val to canonical deterministic key and normalized value.
        """
        if unit_type == "area":
            return "area_ha", val
        elif unit_type == "area_m2":
            # Convert m² to ha for comparison if area_ha is the primary key
            return "area_m2", val
        elif unit_type == "area_km2":
            return "area_km2", val
        elif unit_type == "percentage":
            return "change_pct", val
        elif unit_type in ["distance_m", "distance_km"]:
            return "distance_m", val if unit_type == "distance_m" else val * 1000.0
        return "unknown", val

    def verify(
        self,
        text: str,
        deterministic_numbers: Dict[str, float],
        geojson_geometry: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[GuardrailViolation], str]:
        """
        Verifies all numbers and coordinates in text against ground truth deterministic metrics.
        Returns: (passed: bool, violations: list[GuardrailViolation], rectified_text: str)
        """
        violations: List[GuardrailViolation] = []
        rectified_text = text

        # 1. Number verification
        extractions = self.extract_numbers_with_units(text)
        
        # Categorize deterministic keys by type to support order-of-mention matching
        area_keys = [k for k in deterministic_numbers.keys() if "area" in k or "ha" in k]
        pct_keys = [k for k in deterministic_numbers.keys() if "pct" in k or "percent" in k]
        dist_keys = [k for k in deterministic_numbers.keys() if "distance" in k or "dist" in k]

        area_idx, pct_idx, dist_idx = 0, 0, 0

        for val, u_type, raw_match in extractions:
            matched_key = None
            expected_val = None

            if u_type in ["area", "area_m2", "area_km2"]:
                if u_type == "area" and "area_ha" in deterministic_numbers:
                    matched_key = "area_ha"
                    expected_val = deterministic_numbers["area_ha"]
                elif u_type == "area_m2" and "area_m2" in deterministic_numbers:
                    matched_key = "area_m2"
                    expected_val = deterministic_numbers["area_m2"]
                elif area_keys and area_idx < len(area_keys):
                    matched_key = area_keys[area_idx]
                    expected_val = deterministic_numbers[matched_key]
                    area_idx += 1
            elif u_type == "percentage":
                if "change_pct" in deterministic_numbers:
                    matched_key = "change_pct"
                    expected_val = deterministic_numbers["change_pct"]
                elif pct_keys and pct_idx < len(pct_keys):
                    matched_key = pct_keys[pct_idx]
                    expected_val = deterministic_numbers[matched_key]
                    pct_idx += 1
            elif u_type in ["distance_m", "distance_km"]:
                if dist_keys and dist_idx < len(dist_keys):
                    matched_key = dist_keys[dist_idx]
                    expected_val = deterministic_numbers[matched_key]
                    dist_idx += 1

            if matched_key is None or expected_val is None:
                # Ungrounded number claim
                violations.append(GuardrailViolation(
                    violation_type="ungrounded_claim",
                    raw_claim=raw_match,
                    expected_value=None,
                    actual_value=val,
                    message=f"No deterministic tool metric found for claimed quantity '{raw_match}'."
                ))
                continue

            # Compare within tolerance
            if expected_val == 0.0:
                rel_err = abs(val - expected_val)
            else:
                rel_err = abs(val - expected_val) / abs(expected_val)

            if rel_err > self.tolerance:
                violations.append(GuardrailViolation(
                    violation_type="numerical_drift",
                    raw_claim=raw_match,
                    expected_value=expected_val,
                    actual_value=val,
                    message=f"Claimed '{raw_match}' differs from deterministic '{matched_key}' ({expected_val:.2f}) by {rel_err*100:.1f}% (tolerance +/-{self.tolerance*100:.0f}%)."
                ))
                # Rectify in output text: substitute exact verified number
                replacement = f"{expected_val:.2f} {raw_match.split()[-1]}"
                rectified_text = rectified_text.replace(raw_match, replacement)

        # 2. Coordinate verification against GeoJSON geometry (Point-in-Polygon)
        coords = self.extract_coordinates(text)
        if coords and geojson_geometry:
            try:
                geom = shape(geojson_geometry)
                for lat, lon, raw_coord in coords:
                    # GeoJSON is (lon, lat)
                    pt = Point(lon, lat)
                    if not (geom.contains(pt) or geom.touches(pt)):
                        violations.append(GuardrailViolation(
                            violation_type="spatial_drift",
                            raw_claim=raw_coord,
                            expected_value="Inside detected polygon bounds",
                            actual_value=f"({lat}, {lon})",
                            message=f"Claimed coordinate {raw_coord} lies outside detected spatial evidence polygon."
                        ))
            except Exception as e:
                # Malformed geometry handling
                violations.append(GuardrailViolation(
                    violation_type="spatial_error",
                    raw_claim="GeoJSON",
                    expected_value="Valid geometry",
                    actual_value=str(e),
                    message=f"Unable to parse GeoJSON geometry: {e}"
                ))

        passed = (len(violations) == 0)
        return passed, violations, rectified_text
