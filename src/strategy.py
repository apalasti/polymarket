import abc
import math

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from .transaction import Transaction


class BidOrAsk(NamedTuple):
    size: int
    price: float


@dataclass
class OrderBook:
    bids: list[BidOrAsk]
    asks: list[BidOrAsk]

    def get_ask(self, i: int) -> BidOrAsk:
        if i < len(self.asks):
            return self.asks[i]
        return BidOrAsk(size=0, price=math.nan)

    def get_bid(self, i: int) -> BidOrAsk:
        if i < len(self.bids):
            return self.bids[i]
        return BidOrAsk(size=0, price=math.nan)

    # TODO: Desired price implementation is not the best it should use
    # `total_cost / available` as the max desired price
    def get_liquidity(
        self, desired_shares: int, desired_price: float | None = None
    ) -> tuple[int, float]:
        available = 0
        total_cost = 0.0

        for ask in self.asks:
            if available >= desired_shares:
                break
            if ask.size == 0:
                continue
            if desired_price is not None and ask.price > desired_price:
                continue
            shares_to_buy = min(desired_shares - available, ask.size)
            total_cost += shares_to_buy * ask.price
            available += shares_to_buy

        if available == 0:
            return 0, 0.0

        return available, total_cost / available


@dataclass
class MarketState:
    slug: str
    orderbooks: dict[str, OrderBook]

    @classmethod
    def from_orderbook_snapshot(cls, snapshot: pd.DataFrame) -> "MarketState":
        """Build a MarketState from orderbook table rows for one snapshot (same slug, same ts)."""
        if snapshot.empty:
            raise ValueError("snapshot must not be empty")

        orderbooks: dict[str, OrderBook] = {}
        for _, row in snapshot.iterrows():
            outcome = str(row["outcome"])
            if outcome in orderbooks:
                continue

            bids = [
                BidOrAsk(
                    int(0 if pd.isna(row[f"bid_{i}_size"]) else row[f"bid_{i}_size"]),
                    float(row[f"bid_{i}_price"])
                    if not pd.isna(row[f"bid_{i}_price"])
                    else math.nan,
                )
                for i in range(1, 6)
            ]
            asks = [
                BidOrAsk(
                    int(0 if pd.isna(row[f"ask_{i}_size"]) else row[f"ask_{i}_size"]),
                    float(row[f"ask_{i}_price"])
                    if not pd.isna(row[f"ask_{i}_price"])
                    else math.nan,
                )
                for i in range(1, 6)
            ]
            orderbooks[outcome] = OrderBook(bids=bids, asks=asks)

        slug = str(snapshot["slug"].iloc[0])
        return cls(slug=slug, orderbooks=orderbooks)


class StrategyBase(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, market_state: MarketState, t: int, *args, **kwds
    ) -> list[Transaction]: ...
