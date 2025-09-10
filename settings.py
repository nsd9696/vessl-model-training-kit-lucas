from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = "sk-proj-1234567890"
    gemini_api_key: str = "..."

    class Config:
        env_file = ".env"  # Optional: load from .env file
        env_file_encoding = "utf-8"


def load_settings():
    settings = Settings()
    return settings


if __name__ == "__main__":
    settings = load_settings()
