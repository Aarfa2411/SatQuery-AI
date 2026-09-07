from typing import Any

from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from app.schemas import BoundingBox, ToolOutput


class OutputSanityChecker:
    """
    GeoValidator & Output Sanity Checker:
    Enforces geometric, spatial and confidence integrity on specialist tool outputs.
    """
    def sanitize_bboxes(self, bboxes: list[Any] | None) -> tuple[list[BoundingBox], list[str]]:
        sanitized: list[BoundingBox] = []
        warnings: list[str] = []
        if not bboxes:
            return sanitized, warnings

        for i, bb in enumerate(bboxes):
            if isinstance(bb, dict):
                label = bb.get("label", "Object")
                box_vals = bb.get("box", [0.0, 0.0, 1.0, 1.0])
                score_val = bb.get("score", 1.0)
            else:
                label = bb.label
                box_vals = bb.box
                score_val = bb.score

            ymin, xmin, ymax, xmax = box_vals
            clamped_ymin = max(0.0, min(1.0, float(ymin)))
            clamped_xmin = max(0.0, min(1.0, float(xmin)))
            clamped_ymax = max(0.0, min(1.0, float(ymax)))
            clamped_xmax = max(0.0, min(1.0, float(xmax)))

            if (clamped_ymin != ymin or clamped_xmin != xmin or 
                clamped_ymax != ymax or clamped_xmax != xmax):
                warnings.append(f"Bounding box {i} ('{label}') was out-of-bounds and clamped to [0.0, 1.0].")

            if clamped_ymin >= clamped_ymax or clamped_xmin >= clamped_xmax:
                warnings.append(f"Bounding box {i} ('{label}') had zero or inverted area; skipped.")
                continue

            clamped_score = max(0.0, min(1.0, float(score_val)))
            sanitized.append(BoundingBox(
                label=label,
                box=[round(clamped_ymin, 4), round(clamped_xmin, 4), round(clamped_ymax, 4), round(clamped_xmax, 4)],
                score=round(clamped_score, 4)
            ))
        return sanitized, warnings

    def sanitize_geojson(self, mask_geojson: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
        warnings: list[str] = []
        if not mask_geojson:
            return None, warnings

        try:
            features = mask_geojson.get("features", [])
            valid_features = []
            for i, feat in enumerate(features):
                geom = feat.get("geometry")
                if not geom:
                    continue
                poly = shape(geom)
                if not poly.is_valid:
                    poly = make_valid(poly)
                    warnings.append(f"GeoJSON feature {i} had self-intersections and was corrected via make_valid.")
                
                if poly.is_empty:
                    continue

                valid_features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": feat.get("properties", {})
                })

            sanitized_geojson = {
                "type": "FeatureCollection",
                "features": valid_features
            }
            return sanitized_geojson, warnings
        except Exception as e:
            warnings.append(f"Invalid GeoJSON structure: {e!s}")
            return None, warnings

    def sanitize_tool_output(self, output: ToolOutput) -> ToolOutput:
        all_warnings = list(output.warnings)

        # Sanitize bounding boxes
        clean_bboxes, box_warnings = self.sanitize_bboxes(output.bboxes)
        all_warnings.extend(box_warnings)

        # Sanitize GeoJSON
        clean_geojson, geo_warnings = self.sanitize_geojson(output.mask_geojson)
        all_warnings.extend(geo_warnings)

        # Clamp confidence
        clamped_conf = max(0.05, min(1.0, float(output.confidence)))

        return ToolOutput(
            tool=output.tool,
            execution_mode=output.execution_mode,
            model_version=output.model_version,
            answer=output.answer,
            bboxes=clean_bboxes if clean_bboxes else None,
            mask_geojson=clean_geojson,
            metrics=output.metrics,
            confidence=round(clamped_conf, 4),
            warnings=all_warnings,
            latency_ms=output.latency_ms
        )

output_sanity_checker = OutputSanityChecker()
