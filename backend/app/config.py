from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "SatQuery AI"
    VERSION: str = "3.0.0"
    API_V1_STR: str = "/api"
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    ALIGNED_DIR: Path = STORAGE_DIR / "aligned"
    OUTPUTS_DIR: Path = STORAGE_DIR / "outputs"
    EXPORTS_DIR: Path = STORAGE_DIR / "exports"
    DB_PATH: Path = STORAGE_DIR / "sessions.db"
    
    # Security and Storage Caps
    MAX_FILE_SIZE_BYTES: int = 250 * 1024 * 1024  # 250 MB
    MAX_SESSION_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB
    SESSION_TTL_HOURS: int = 2
    SECRET_KEY: str = "satquery-isro-sih26167-secret-key-2026"
    
    # Execution & Device settings
    ALLOW_GPU: bool = True
    DEFAULT_CRS: str = "EPSG:4326"
    TARGET_OPTICAL_RES_M: float = 0.65  # Cartosat-2S native GSD (m)
    
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()

# Ensure directories exist
for directory in [settings.UPLOADS_DIR, settings.ALIGNED_DIR, settings.OUTPUTS_DIR, settings.EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
