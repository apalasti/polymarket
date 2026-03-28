import math
from scipy.stats import norm

from .strategy import ExecutorProtocol, MarketState, StrategyBase
from .transaction import OrderType, Transaction

MAX_POSITION_SIZE = 15
EDGE_THRESHOLD_BUY = 0.23
EDGE_THRESHOLD_SELL = 0.23
VOLATILITY = 0.00002
MAX_KELLY_FRACTION = 0.003
MAX_T = 780


class AIStrategy(StrategyBase):
    def __init__(self, executor: ExecutorProtocol):
        super().__init__(executor=executor)
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

        if t > MAX_T:
            return []

        start_price = self.start_prices[slug]

        ret = math.log(price / start_price) if start_price > 0 else 0

        time_remaining = 900 - t
        time_fraction = time_remaining / 900.0

        if time_fraction <= 0:
            return []

        current_vol = VOLATILITY * math.sqrt(time_fraction)

        if current_vol > 0:
            prob_up = norm.cdf(ret / current_vol)
        else:
            prob_up = 1.0 if price > start_price else 0.0

        prob_up = max(0.01, min(0.99, prob_up))
        prob_down = 1.0 - prob_up

        transactions = []

        for outcome in ["Up", "Down"]:
            ob = market_state.orderbooks.get(outcome)
            if not ob:
                continue

            prob = prob_up if outcome == "Up" else prob_down

            ask_1 = ob.get_ask(0)
            if not math.isnan(ask_1.price) and ask_1.price > 0:
                if ask_1.price < prob - EDGE_THRESHOLD_BUY:
                    desired_price = ask_1.price
                    pos = self.executor.get_positions(slug).get(outcome, 0)
                    if pos < MAX_POSITION_SIZE:
                        edge = prob - ask_1.price
                        kelly = edge / (1.0 - ask_1.price)
                        fraction = max(0.0, min(MAX_KELLY_FRACTION, kelly))
                        capital_to_risk = self.executor.capital * fraction
                        shares_to_buy = int(capital_to_risk / desired_price)
                        shares_to_buy = int(
                            min(shares_to_buy, MAX_POSITION_SIZE - pos, ask_1.size)
                        )

                        if shares_to_buy > 0:
                            available, execute_price = ob.get_liquidity(
                                shares_to_buy, desired_price
                            )
                            if (
                                available > 0
                                and (execute_price * available) <= self.executor.capital
                            ):
                                transactions.append(
                                    Transaction(
                                        slug=slug,
                                        outcome=outcome,
                                        order_type=OrderType.BUY,
                                        shares=available,
                                        price=execute_price,
                                        t=t,
                                    )
                                )

            bid_1 = ob.get_bid(0)
            if not math.isnan(bid_1.price) and bid_1.price > 0:
                pos = self.executor.get_positions(slug).get(outcome, 0)
                if pos > 0:
                    if bid_1.price > prob + EDGE_THRESHOLD_SELL:
                        min_price = bid_1.price
                        shares_to_sell = int(min(pos, bid_1.size))

                        filled, execute_price = ob.get_sell_liquidity(
                            shares_to_sell, min_price
                        )
                        if filled > 0:
                            transactions.append(
                                Transaction(
                                    slug=slug,
                                    outcome=outcome,
                                    order_type=OrderType.SELL,
                                    shares=filled,
                                    price=min_price,
                                    t=t,
                                )
                            )

        return transactions
