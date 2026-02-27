"""15m crypto market helpers."""

from dataclasses import dataclass
import logging
from datetime import datetime, timedelta, timezone
import math

import requests

from src.fetch import get_markets_by_slug, _to_utc_ts
from src.settings import settings
from src.strategy import MarketState, StrategyBase
from src.transaction import OrderType, Transaction


SLUG_BATCH_SIZE = 50
INTERVAL_15M = 900

logger = logging.getLogger(__name__)


@dataclass
class CryptoMarkets15m:
    asset: str
    start: datetime
    end: datetime

    def __str__(self) -> str:
        return f"CryptoMarkets15m(asset={self.asset!r}, start={self.start.strftime('%Y-%m-%d %H:%M')}, end={self.end.strftime('%Y-%m-%d %H:%M')})"

    def generate_slugs(self) -> list[str]:
        """Generate 15m market slugs for this asset from start to end (aligned to 900s)."""

        start_aligned = (_to_utc_ts(self.start) // INTERVAL_15M) * INTERVAL_15M
        end_aligned = (_to_utc_ts(self.end) // INTERVAL_15M) * INTERVAL_15M

        slugs = []
        for ts in range(start_aligned, end_aligned + 1, INTERVAL_15M):
            slugs.append(f"{self.asset}-updown-15m-{ts}")

        return slugs

    def iter_markets(self):
        """Yield each market dict once from Gamma (batched by slug)."""

        slugs = self.generate_slugs()
        for i in range(0, len(slugs), SLUG_BATCH_SIZE):
            batch = slugs[i : i + SLUG_BATCH_SIZE]

            try:
                markets = get_markets_by_slug(batch)
                logger.debug(
                    "Fetched %d markets for batch %d/%d slugs: [%s, ...]",
                    len(markets),
                    i + 1,
                    math.ceil(len(slugs) / SLUG_BATCH_SIZE),
                    batch[0],
                )
            except Exception as e:
                logger.warning("Fetching markets failed for slugs %s: %s", batch[:3], e)
                continue

            for m in markets:
                yield m

    def get_prices(
        self,
        start: datetime,
        end: datetime,
        *,
        interval: str = "15m",
    ) -> list[float]:
        """Fetch Binance USDT close price per kline between start and end. interval is Binance kline interval (e.g. '1m', '15m')."""
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        symbol = f"{self.asset.upper()}USDT"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 900,
        }
        response = requests.get(settings.BINANCE_URL, params=params)
        data = response.json()

        # kline: [open_time, open, high, low, close, volume, ...]
        return [
            float(candle[4])
            for candle in data
        ]

    @staticmethod
    def slug_to_time_range(slug: str) -> tuple[datetime, datetime] | None:
        """
        Given a slug like 'btc-updown-15m-1700000000', return (start, end) as UTC datetimes.
        Returns None if the slug format does not match.
        """
        try:
            parts = slug.split("-")
            if len(parts) < 4 or parts[-2] != "15m":
                return None

            period_ts = int(parts[-1])
            start = datetime.fromtimestamp(period_ts, tz=timezone.utc)
            end = start + timedelta(seconds=INTERVAL_15M)
            return start, end
        except Exception:
            return None


class PriceDirectionStrategy(StrategyBase):
    """
    At a given time t: buy Up if price(t) > price(0), else buy Down, only if the
    chosen outcome's ask is less then the predicted probability.
    """

    def __init__(self):
        self.initial_price = None
        self.initial_market = None

    def __call__(
        self, market_state: MarketState, t: int, price: float,
    ) -> Transaction | None:
        if self.initial_market != market_state.slug:
            self.initial_price = price
            self.initial_market = market_state.slug

        if t != 800:
            return None

        outcome_to_buy = "Up" if self.initial_price < price else "Down"
        best_ask = market_state.orderbooks[outcome_to_buy].get_ask(0)
        if best_ask.price < 0.92:
            return Transaction(
                outcome=outcome_to_buy,
                order_type=OrderType.BUY,
                shares=1,
                price=best_ask.price,
            )

        return None
