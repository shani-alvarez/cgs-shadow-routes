import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_ROOT: Path = Path(os.getenv("DATA_ROOT", Path.cwd() / "data"))
    RAW: Path = DATA_ROOT / "raw"
    INTERIM: Path = DATA_ROOT / "interim"
    PROCESSED: Path = DATA_ROOT / "processed"
    OUTPUTS: Path = Path.cwd() / "outputs"

    class Config:
        env_file = ".env"


settings = Settings()
for p in [settings.RAW, settings.INTERIM, settings.PROCESSED, settings.OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)
