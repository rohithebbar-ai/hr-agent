"""
Settings loader — picks the right config based on ENVIRONMENT variable.

Usage:
    from config.settings import settings
    print(settings.ENVIRONMENT)
    print(settings.QDRANT_URL)
"""

import os
from pathlib import Path
from dotenv import load_dotenv


ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
CONFIG_DIR = Path(__file__).parent

# Load environment-specific config
env_file = CONFIG_DIR / f"{ENVIRONMENT}.env"
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"[CONFIG] Loaded {ENVIRONMENT} environment from {env_file}")
else:
    print(f"[CONFIG] No config file for {ENVIRONMENT}, using defaults")


class Settings:
    ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
    DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

    # Infrastructure
    QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Rate limiting
    RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")

    # Caching
    CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))

    # Evaluation
    EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "50"))

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "prod"

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT == "dev"


settings = Settings()