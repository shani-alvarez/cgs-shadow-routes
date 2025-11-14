from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Path to repo
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    # Tell pydantic-settings exactly where the .env is
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default if .env is missing: repo/data
    DATA_ROOT: Path = PROJECT_ROOT / "data"

    @property
    def RAW(self) -> Path:
        return self.DATA_ROOT / "raw"

    @property
    def INTERIM(self) -> Path:
        return self.DATA_ROOT / "interim"

    @property
    def PROCESSED(self) -> Path:
        return self.DATA_ROOT / "processed"

    @property
    def OUTPUTS(self) -> Path:
        return PROJECT_ROOT / "outputs"


settings = Settings()

# Make sure dirs exist (optional)
for p in [settings.RAW, settings.INTERIM, settings.PROCESSED, settings.OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)
