"""
Geospatial Precision Metrics
Validates vector geometries, polygon boundaries, and coordinate alignments.
"""

import math
from typing import Dict, Any, Union
from shapely.geometry import shape, Point, Polygon


def compute_polygon_iou(
    pred_geom: Union[Dict[str, Any], Polygon],
    gt_geom: Union[Dict[str, Any], Polygon]
) -> float:
    """
    Computes spatial Intersection over Union (IoU) between two Shapely polygons / GeoJSON geometries.
    """
    try:
        p = shape(pred_geom) if isinstance(pred_geom, dict) else pred_geom
        g = shape(gt_geom) if isinstance(gt_geom, dict) else gt_geom

        if not p.is_valid:
            p = p.buffer(0)
        if not g.is_valid:
            g = g.buffer(0)

        inter = p.intersection(g).area
        union = p.union(g).area

        if union == 0:
            return 1.0 if inter == 0 else 0.0
        return float(inter / union)
    except Exception:
        return 0.0


def compute_centroid_distance_meters(
    pred_geom: Union[Dict[str, Any], Polygon],
    gt_geom: Union[Dict[str, Any], Polygon],
    is_wgs84: bool = True
) -> float:
    """
    Calculates geographic offset distance between polygon centroids in meters.
    Uses Haversine approximation if coordinates are in WGS84 (lon, lat).
    """
    try:
        p = shape(pred_geom) if isinstance(pred_geom, dict) else pred_geom
        g = shape(gt_geom) if isinstance(gt_geom, dict) else gt_geom

        c_p = p.centroid
        c_g = g.centroid

        if not is_wgs84:
            # Planar / UTM projected coordinates (already in meters)
            return float(math.hypot(c_p.x - c_g.x, c_p.y - c_g.y))

        # Haversine distance for WGS84 coordinates (lon, lat)
        lon1, lat1 = math.radians(c_p.x), math.radians(c_p.y)
        lon2, lat2 = math.radians(c_g.x), math.radians(c_g.y)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        radius_m = 6371000.0  # Earth mean radius in meters

        return float(radius_m * c)
    except Exception:
        return float("inf")


def point_in_polygon_test(lon: float, lat: float, geom: Union[Dict[str, Any], Polygon]) -> bool:
    """
    Tests whether a (lon, lat) coordinate falls strictly within or on the boundary of a geometry.
    """
    try:
        g = shape(geom) if isinstance(geom, dict) else geom
        pt = Point(lon, lat)
        return bool(g.contains(pt) or g.touches(pt))
    except Exception:
        return False
