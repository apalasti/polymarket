"""Transaction execution validation and evaluation."""

from dataclasses import dataclass

from .strategy import MarketState
from .transaction import OrderType, Transaction


class ExecutionError(Exception):
    """Raised when a transaction cannot be executed."""

    def __init__(self, transaction: Transaction, reason: str):
        self.transaction = transaction
        self.reason = reason
        super().__init__(f"Transaction cannot be executed: {transaction}, reason: {reason}")


class Executor:
    """Owns capital and positions; validates and evaluates transactions against a market state."""

    def __init__(self, capital: float):
        self._capital = capital
        self._positions_by_slug: dict[str, dict[str, int]] = {}

    @property
    def capital(self) -> float:
        return self._capital

    def get_positions(self, slug: str) -> dict[str, int]:
        """Return current positions (outcome -> shares) for the given market."""
        return self._positions_by_slug.setdefault(slug, {})

    def portfolio_value(self, market_prices: dict[str, dict[str, float]]) -> float:
        """Total value of capital plus all positions at the given prices.

        market_prices: slug -> outcome -> price (e.g. mid or last trade).
        Missing (slug, outcome) prices are treated as 0.
        """
        total = self._capital
        for slug, positions in self._positions_by_slug.items():
            prices = market_prices.get(slug, {})
            for outcome, shares in positions.items():
                total += shares * prices.get(outcome, 0.0)
        return total

    def execute(
        self,
        transactions: list[Transaction],
        market_state: MarketState,
    ):
        """Check all transactions can be executed; if so, apply them and update state. Raise ExecutionError on first invalid transaction."""
        slug = market_state.slug
        pos = dict(self.get_positions(slug))
        cap = self._capital
        executed: list[Transaction] = []

        for tx in transactions:
            if tx.slug != market_state.slug:
                raise ExecutionError(tx, "transaction slug does not match market")

            if tx.order_type == OrderType.SELL:
                held = pos.get(tx.outcome, 0)
                if held < tx.shares:
                    raise ExecutionError(tx, "insufficient position")
                orderbook = market_state.orderbooks.get(tx.outcome)
                if orderbook is None:
                    raise ExecutionError(tx, "no orderbook for outcome")
                filled, _ = orderbook.get_sell_liquidity(tx.shares, tx.price)
                if filled < tx.shares:
                    raise ExecutionError(tx, "insufficient bid liquidity")
                pos[tx.outcome] = held - tx.shares
                cap += tx.shares * tx.price
                executed.append(tx)
                continue

            # BUY: tx.price is average fill price from strategy, not a max price cap.
            # Only check that the book has enough size to fill the order.
            orderbook = market_state.orderbooks.get(tx.outcome)
            if orderbook is None:
                raise ExecutionError(tx, "no orderbook for outcome")
            available, _ = orderbook.get_liquidity(tx.shares, None)
            if available < tx.shares:
                raise ExecutionError(tx, "insufficient liquidity")
            cost = tx.cost()
            if cap < cost:
                raise ExecutionError(tx, "insufficient capital")
            pos[tx.outcome] = pos.get(tx.outcome, 0) + tx.shares
            cap -= cost
            executed.append(tx)

        self._positions_by_slug[slug] = pos
        self._capital = cap
