"""satquery.core — shared geospatial utilities."""
from .raster_io import RasterData, load_raster, load_raster_from_array, match_raster, save_mask, save_raster, save_geojson, mask_to_geojson
from .validator import ValidationError, validate_input_pair, check_polygons_in_bounds

__all__ = [
    "RasterData", "load_raster", "load_raster_from_array", "match_raster",
    "save_mask", "save_raster", "save_geojson", "mask_to_geojson",
    "ValidationError", "validate_input_pair", "check_polygons_in_bounds",
]
