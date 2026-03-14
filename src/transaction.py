from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Transaction:
    slug: str
    outcome: str
    order_type: OrderType
    shares: int
    price: float

    t: datetime | int
