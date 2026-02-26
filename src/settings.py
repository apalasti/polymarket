import pathlib

from dome_api_sdk import DomeClient
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    DOME_API_KEY: str

    DATA_API_URL: str = Field(default="https://data-api.polymarket.com")
    GAMMA_API_URL: str = Field(default="https://gamma-api.polymarket.com")
    CLOB_API_URL: str = Field(default="https://clob.polymarket.com")
    BINANCE_URL: str = Field(default="https://api.binance.com/api/v3/klines")

    DATA_DIR: pathlib.Path = Field(
        default=pathlib.Path(__file__).parent.parent / "data"
    )

    def dome_client(self) -> DomeClient:
        if hasattr(self, "_dome_client"):
            return self._dome_client
        self._dome_client = DomeClient({"api_key": self.DOME_API_KEY})
        return self._dome_client


load_dotenv()
settings = Settings()
