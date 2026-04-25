import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Task Manager API")
    db_url: str = os.getenv("DB_URL", "sqlite:///./tasks.db")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    api_key: str = os.getenv("API_KEY", "default_secret_key")

settings = Settings()
