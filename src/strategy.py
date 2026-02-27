import abc
import math

from dataclasses import dataclass
from typing import NamedTuple

from src.transaction import Transaction


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


@dataclass
class MarketState:
    slug: str
    orderbooks: dict[str, OrderBook]


class StrategyBase(abc.ABC):

    @abc.abstractmethod
    def __call__(
        self, market_state: MarketState, t: int, *args, **kwds
    ) -> Transaction | None: ...
