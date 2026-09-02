import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from app.config import settings
from app.services.geo_compat import HAS_RASTERIO

def create_sample_geotiffs():
    samples_dir = settings.BASE_DIR / "sample_data"
    samples_dir.mkdir(parents=True, exist_ok=True)

    opt1_path = samples_dir / "cartosat2s_optical_20240301.tif"
    opt2_path = samples_dir / "cartosat2s_optical_20240901.tif"
    sar_path = samples_dir / "risat1a_sar_20240901.tif"

    w, h = 256, 256

    # Optical T1: Rural & Forest baseline
    opt1_data = np.random.randint(60, 180, (h, w, 3), dtype=np.uint8)
    opt1_data[60:140, 60:140, :] = [30, 80, 140] # Water body

    # Optical T2: Expanded urban settlement
    opt2_data = opt1_data.copy()
    opt2_data[150:220, 150:220, :] = [230, 220, 210] # High-reflectance new built-up area

    # SAR: Microwave backscatter
    sar_data = np.random.randint(20, 120, (h, w), dtype=np.uint8)
    sar_data[60:140, 60:140] = 10 # Low backscatter specular water
    sar_data[150:220, 150:220] = 240 # Double-bounce high backscatter urban structures

    if HAS_RASTERIO:
        import rasterio
        from rasterio.transform import from_origin
        transform = from_origin(500000, 3000000, 0.65, 0.65)
        for p, d, c in [(opt1_path, np.transpose(opt1_data, (2, 0, 1)), 3),
                        (opt2_path, np.transpose(opt2_data, (2, 0, 1)), 3),
                        (sar_path, np.expand_dims(sar_data, axis=0), 1)]:
            with rasterio.open(
                p, 'w', driver='GTiff', height=h, width=w, count=c,
                dtype=np.uint8, crs='EPSG:32643', transform=transform
            ) as dst:
                dst.write(d)
    else:
        Image.fromarray(opt1_data).save(opt1_path)
        Image.fromarray(opt2_data).save(opt2_path)
        Image.fromarray(sar_data).save(sar_path)

    print(f"[Sample Data] Generated sample GeoTIFFs in {samples_dir}:")
    print(f"  - {opt1_path.name}")
    print(f"  - {opt2_path.name}")
    print(f"  - {sar_path.name}")

if __name__ == "__main__":
    create_sample_geotiffs()
