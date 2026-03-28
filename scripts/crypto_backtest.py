import argparse
import os
import pathlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from src.crypto_market import CryptoMarkets15m
from src.ai_strategy import AIStrategy
from src.execution import Executor
from src.strategy import MarketState


INITIAL_CAPITAL = 500
CRYPTO_DATA_DIR = pathlib.Path(__file__).parent.parent / "data/crypto-prices"
DB_PATH = CRYPTO_DATA_DIR / "test.duckdb"


def worker(task_id: str, slugs: list[str], initial_capital: int, progress: dict):
    slugs = sorted(slugs)

    capital = initial_capital
    returns = []
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        slugs_df = pd.DataFrame({"slug": slugs})
        conn.register("slugs_df", slugs_df)
        all_df = conn.execute("""
            SELECT o.*, cp.price, r.outcome_price
            FROM orderbook o
            JOIN slugs_df s ON o.slug = s.slug
            JOIN crypto_price cp ON o.slug = cp.slug
                AND o.secs_from_start = cp.secs_from_start
            JOIN resolved r ON o.slug = r.slug AND r.outcome = 'Up'
            ORDER BY o.slug, o.ts, o.secs_from_start
        """).fetchdf()

        grouped = all_df.groupby("slug", sort=False)
        for i, slug in enumerate(slugs):
            if slug not in grouped.groups:
                returns.append(1.0)
                continue

            df = grouped.get_group(slug)

            executor = Executor(capital=capital)
            strategy = AIStrategy(executor=executor)
            for _, snapshot in df.groupby(["slug", "ts"]):
                state = MarketState.from_orderbook_snapshot(snapshot)
                strategy.process_orderbook(
                    state,
                    t=int(snapshot["secs_from_start"].iloc[-1]),
                    price=float(snapshot["price"].iloc[-1]),
                )

            outcome_price = float(df["outcome_price"].iloc[0])
            new_capital = executor.portfolio_value(
                {slug: {"Up": outcome_price, "Down": 1 - outcome_price}}
            )

            returns.append(new_capital / capital)

            capital = new_capital

            progress[task_id] = {
                "completed": i + 1,
                "total": len(slugs),
                "slug": slug,
                "capital": capital,
            }

    return np.array(returns)


def calcultate_statistics(returns: np.ndarray):
    period_returns = returns - 1.0
    n = len(period_returns)
    mean_r = float(np.mean(period_returns))
    std_r = float(np.std(period_returns))
    sharpe = (mean_r / std_r * np.sqrt(n)) if std_r > 0 else 0.0
    win_rate = float(np.mean(period_returns > 0)) if n else 0.0

    cum_gross = np.cumprod(returns)
    total_return = cum_gross[-1] - 1.0 if n else 0.0

    # Max drawdown from cumulative gross
    peak = np.maximum.accumulate(cum_gross)
    drawdown = (cum_gross - peak) / peak
    max_dd = float(np.min(drawdown)) if n else 0.0

    return {
        "markets": n,
        "markets_traded": int((period_returns == 0).sum()),
        "total_return": total_return,
        "mean_period_return": mean_r,
        "volatility": std_r,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }


def plot_returns(stats: dict):
    series = {
        f"{label} | Wins: {data['win_rate'] * 100:.2f}% | Sharpe: {data['sharpe_ratio']:.2f} | Drawdown: {data['max_drawdown'] * 100:.2f}%": np.cumprod(
            data["returns"]
        )
        for label, data in stats.items()
    }

    max_len = max(len(s) for s in series.values())
    df = pd.DataFrame(
        {
            k: np.pad(s, (0, max_len - len(s)), constant_values=np.nan)
            for k, s in series.items()
        }
    )
    fig = px.line(
        df,
        labels={"value": "Cumulative return", "index": "Market index"},
    )
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        height=800,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2,
            yanchor="top",
        ),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Run crypto backtest (optionally in parallel over time periods)."
    )
    parser.add_argument(
        "-n",
        "--periods",
        type=int,
        default=10,
        help="Number of distinct time periods to run in parallel (default: 10)",
    )
    args = parser.parse_args()
    n_periods = max(1, args.periods)

    with duckdb.connect(DB_PATH, read_only=True) as conn:
        slugs = (
            conn.query("SELECT DISTINCT slug FROM resolved ORDER BY slug;")
            .fetchdf()["slug"]
            .tolist()
        )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        chunks = np.array_split(slugs, n_periods)
        print("Num markets for each period", list(map(len, chunks)))

        with multiprocessing.Manager() as manager:
            progress_dict = manager.dict()
            overall_progress_task = progress.add_task("[green]All markets progress:")

            with ProcessPoolExecutor(
                max_workers=min(n_periods, os.cpu_count() - 1),
            ) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    start, _ = CryptoMarkets15m.slug_to_time_range(chunk[0])
                    _, end = CryptoMarkets15m.slug_to_time_range(chunk[-1])

                    task_id = progress.add_task(f"Period {i}: {start.date()} - {end.date()}", visible=False)
                    futures.append(
                        executor.submit(worker, task_id, chunk, INITIAL_CAPITAL, progress_dict)
                    )

                # monitor the progress:
                while (n_finished := sum([future.done() for future in futures])) < len(
                    futures
                ):
                    progress.update(
                        overall_progress_task, completed=n_finished, total=len(futures)
                    )
                    for task_id, update_data in progress_dict.items():
                        latest = update_data["completed"]
                        total = update_data["total"]
                        progress.update(
                            task_id,
                            completed=latest,
                            total=total,
                            visible=latest < total,
                        )

                returns = [future.result() for future in futures]

    stats = {}
    for i in range(n_periods):
        start, _ = CryptoMarkets15m.slug_to_time_range(chunks[i][0])
        _, end = CryptoMarkets15m.slug_to_time_range(chunks[i][-1])

        stats[f"{start.date()} - {end.date()}"] = {
            "returns": returns[i],
        } | calcultate_statistics(returns[i])

    # Average summary of statistics across all periods
    stats_values = list(stats.values())
    avg_total_return = np.nanmean([v.get("total_return", np.nan) for v in stats_values])
    avg_mean_return = np.nanmean(
        [v["mean_period_return"] for v in stats_values if "mean_period_return" in v]
    )
    avg_volatility = np.nanmean(
        [v["volatility"] for v in stats_values if "volatility" in v]
    )
    avg_sharpe = np.nanmean(
        [v["sharpe_ratio"] for v in stats_values if "sharpe_ratio" in v]
    )
    avg_win_rate = np.nanmean([v["win_rate"] for v in stats_values if "win_rate" in v])
    max_drawdown = np.min(
        [v["max_drawdown"] for v in stats_values if "max_drawdown" in v]
    )
    total_markets = sum([len(v["returns"]) for v in stats_values])
    periods_count = len(stats_values)

    print("\n--- Average Backtest Summary ---")
    print(f"Periods: {periods_count}")
    print(f"Markets: {total_markets}")
    print(f"Total return (mean): {avg_total_return:.2%}")
    print(f"Mean period return (mean): {avg_mean_return:.4%}")
    print(f"Volatility (std, mean): {avg_volatility:.4%}")
    print(f"Sharpe ratio (mean): {avg_sharpe:.2f}")
    print(f"Win rate (mean): {avg_win_rate:.1%}")
    print(f"Max drawdown: {max_drawdown:.2%}")

    fig = plot_returns(stats)
    fig.write_image("cumulative_returns.png", scale=2)
    print("Saved plot to cumulative_returns.png")


if __name__ == "__main__":
    main()
