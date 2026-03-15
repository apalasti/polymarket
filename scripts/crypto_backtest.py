import pathlib

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from src.ai_strategy import AIStrategy
from src.execution import ExecutionError, Executor
from src.strategy import MarketState

CRYPTO_DATA_DIR = pathlib.Path(__file__).parent.parent / "data/crypto-prices"


def main():
    capital = 100

    returns = []
    with duckdb.connect(CRYPTO_DATA_DIR / "test.duckdb") as conn:
        slugs: pd.Series = conn.query("SELECT DISTINCT slug FROM resolved ORDER BY slug;").fetchdf()["slug"]
        slugs = slugs.iloc[:100]
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
    drawdown = (cum_gross - peak) / np.maximum(peak, 1e-12)
    max_dd = float(np.min(drawdown)) if n else 0.0

    print("\n--- Backtest summary ---")
    print(f"Markets: {n}")
    print(f"Total return: {total_return:.2%}")
    print(f"Mean period return: {mean_r:.4%}")
    print(f"Volatility (std): {std_r:.4%}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Max drawdown: {max_dd:.2%}")

    # Plot cumulative returns
    if n > 0:
        plot_df = pd.DataFrame({
            "Market": range(n),
            "Cumulative return": cum_gross,
        })
        fig = px.line(
            plot_df,
            x="Market",
            y="Cumulative return",
        )
        fig.update_layout(
            xaxis_title="Market index",
            yaxis_title="Cumulative gross return",
            # yaxis_tickformat=".2f",
            # margin=dict(t=60, b=50, l=60, r=40),
            # hovermode="x unified",
            showlegend=False,
        )
        fig.write_image("cumulative_returns.png")
        print("Saved cumulative returns plot to cumulative_returns.png")


if __name__ == "__main__":
    main()
