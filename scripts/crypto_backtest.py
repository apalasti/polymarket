import pathlib

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from src.ai_strategy import AIStrategy
from src.execution import ExecutionError, Executor
from src.strategy import MarketState


INITIAL_CAPITAL = 500
CRYPTO_DATA_DIR = pathlib.Path(__file__).parent.parent / "data/crypto-prices"


def main():
    capital = INITIAL_CAPITAL

    returns = []
    with duckdb.connect(CRYPTO_DATA_DIR / "test.duckdb") as conn:
        slugs: pd.Series = conn.query("SELECT DISTINCT slug FROM resolved ORDER BY slug;").fetchdf()["slug"]
        for _, slug in (
            pbar := tqdm(slugs.items(), total=len(slugs), desc="Processing slugs")
        ):
            df = conn.execute("""
                SELECT o.*, cp.price, r.outcome_price FROM orderbook o 
                JOIN crypto_price cp ON o.slug = cp.slug 
                    AND o.secs_from_start = cp.secs_from_start 
                JOIN resolved r ON o.slug = r.slug
                WHERE o.slug = ? AND r.outcome = 'Up'
                ORDER BY o.ts, o.secs_from_start""",
                [slug],
            ).fetchdf()

            executor = Executor(capital=capital)
            strategy = AIStrategy(executor=executor)
            transactions = []
            for _, snapshot in df.groupby(["slug", "ts"]):
                state = MarketState.from_orderbook_snapshot(snapshot)
                try:
                    txs = strategy.process_orderbook(
                        state,
                        t=int(snapshot["secs_from_start"].iloc[-1]),
                        price=float(snapshot["price"].iloc[-1]),
                    )
                except ExecutionError as e:
                    print("Transactions:", transactions)
                    print(executor.capital)
                    raise e
                transactions.extend(txs)

            slug = df["slug"].iloc[0]
            outcome_price = df["outcome_price"].iloc[0]
            new_capital = executor.portfolio_value(
                {slug: {"Up": outcome_price, "Down": 1 - outcome_price}}
            )

            returns.append(new_capital / capital)

            capital = new_capital
            pbar.set_postfix({"capital": capital})

    # Summary statistics
    gross = np.array(returns)
    period_returns = gross - 1.0
    n = len(period_returns)
    cum_gross = np.cumprod(gross)
    total_return = cum_gross[-1] - 1.0 if n else 0.0
    mean_r = float(np.mean(period_returns))
    std_r = float(np.std(period_returns))
    sharpe = (mean_r / std_r * np.sqrt(n)) if std_r > 0 else 0.0
    win_rate = float(np.mean(period_returns > 0)) if n else 0.0
    # Max drawdown from cumulative gross
    peak = np.maximum.accumulate(cum_gross)
    drawdown = (cum_gross - peak) / peak
    max_dd = float(np.min(drawdown)) if n else 0.0

    print("\n--- Backtest summary ---")
    print(f"Markets: {n}")
    print(f"Total return: {total_return:.2%}")
    print(f"Mean period return: {mean_r:.4%}")
    print(f"Volatility (std): {std_r:.4%}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Max drawdown: {max_dd:.2%}")

    # Plot cumulative returns, capital, and drawdown
    if n > 0:
        x = np.arange(n)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Cumulative gross return", "Capital ($)", "Drawdown"),
        )
        fig.add_trace(
            go.Scatter(x=x, y=cum_gross, name="Return", line=dict(color="#2563eb", width=2)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=INITIAL_CAPITAL * cum_gross, name="Capital", line=dict(color="#059669", width=2)),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=drawdown, name="Drawdown", fill="tozeroy", line=dict(color="#dc2626", width=1.5)),
            row=3, col=1,
        )
        fig.update_layout(
            height=800,
            showlegend=False,
        )
        fig.update_yaxes(tickformat=".2f", row=1)
        fig.update_yaxes(tickformat="$.0f", row=2)
        fig.update_yaxes(tickformat=".0%", row=3)
        fig.update_xaxes(title_text="Market index", row=3)
        fig.write_image("cumulative_returns.png", scale=2)
        print("Saved cumulative returns plot to cumulative_returns.png")


if __name__ == "__main__":
    main()
