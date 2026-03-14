import json

import pandas as pd

from .settings import settings
from .strategy import MarketState, StrategyBase
from .transaction import OrderType, Transaction


with open(settings.DATA_DIR / "crypto-prices/prob_state_same.json") as f:
    prob_state_same = pd.DataFrame(json.load(f))


BASE_SHARE_COUNT = 10
MIN_CONFIDENCE_THRESHOLD = 0.05


class AIStrategy(StrategyBase):
    def __init__(self):
        self.start_prices: dict[str, float] = {}

    def __call__(
        self, market_state: MarketState, t: int, price: float
    ) -> list[Transaction]:
        slug = market_state.slug

        if t == 0:
            self.start_prices[slug] = price
            return []

        if slug not in self.start_prices:
            return []

        start_price = self.start_prices[slug]

        prob = self._get_probability(t)

        confidence = abs(prob - 0.5)
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return []

        current_direction = "Up" if price > start_price else "Down"

        if prob > 0.5:
            bet_outcome = current_direction
        else:
            bet_outcome = "Down" if current_direction == "Up" else "Up"

        desired_shares = int(confidence * BASE_SHARE_COUNT * 2)

        orderbook = market_state.orderbooks.get(bet_outcome)
        if not orderbook:
            return []

        available, execute_price = orderbook.get_liquidity(desired_shares, prob - 0.05)
        if available == 0:
            return []

        return [
            Transaction(
                slug=slug,
                outcome=bet_outcome,
                order_type=OrderType.BUY,
                shares=available,
                price=execute_price,
                t=t,
            )
        ]

    def _get_probability(self, t: int) -> float:
        idx = (prob_state_same["secs_from_start"] - t).abs().argsort().iloc[0]
        return float(prob_state_same.iloc[idx]["prob_state_same"])
