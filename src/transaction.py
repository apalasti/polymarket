from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, TypedDict


class OrderType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Transaction:
    outcome: str
    order_type: OrderType
    shares: int
    price: float
