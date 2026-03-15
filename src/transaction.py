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

    def cost(self) -> float:
        """Returns the capital impact of this transaction.

        Positive = capital consumed (money leaves strategy)
        Negative = capital added (money enters strategy)
        """
        if self.order_type == OrderType.BUY:
            return self.shares * self.price
        return -self.shares * self.price
