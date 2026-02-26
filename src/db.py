import json
import logging
from datetime import datetime, timezone

import duckdb

from src.fetch import get_markets, get_trades

logger = logging.getLogger(__name__)


def connect_to_db(path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(path)
    __init_schema(conn)
    return conn


def __init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            -- Unix time when the trade was executed (API: timestamp).
            timestamp TIMESTAMP WITH TIME ZONE,
            -- Market condition ID, Hash64 identifying the market (API: conditionId).
            condition_id VARCHAR,
            -- URL-friendly market identifier (API: slug).
            slug VARCHAR,
            -- Outcome token / asset identifier (API: asset).
            asset VARCHAR,
            -- Trader's proxy wallet address (API: proxyWallet).
            proxy_wallet VARCHAR,
            -- Trade direction: BUY or SELL (API: side).
            side VARCHAR,
            -- Number of shares/contracts traded (API: size).
            size DOUBLE,
            -- Price per share, typically in [0, 1] for binary markets (API: price).
            price DOUBLE,
            -- Human-readable outcome label, e.g. Yes/No (API: outcome).
            outcome VARCHAR,
            -- 0-based index of the outcome in the market (API: outcomeIndex).
            outcome_index INTEGER,
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            -- Unique market identifier (API: id).
            id VARCHAR PRIMARY KEY,
            -- Condition ID, Hash64 identifying the market (API: conditionId).
            condition_id VARCHAR,
            -- Market question text (API: question).
            question VARCHAR,
            -- URL-friendly market identifier (API: slug).
            slug VARCHAR,
            -- Full market description / resolution criteria (API: description).
            description VARCHAR,
            -- JSON array of outcome labels, e.g. ["Yes", "No"] (API: outcomes).
            outcomes VARCHAR,
            -- JSON array of current prices per outcome, e.g. ["0.6", "0.4"] (API: outcomePrices).
            outcome_prices VARCHAR,
            -- JSON array of category names, e.g. ["Politics", "Sports"] (API: categories).
            categories VARCHAR,
            -- Market open / start time (API: startDate).
            start_date TIMESTAMP WITH TIME ZONE,
            -- Market resolution / end time (API: endDate).
            end_date TIMESTAMP WITH TIME ZONE,
            -- Total trading volume in USD (API: volumeNum).
            volume_num DOUBLE,
            -- Whether the market is currently active (API: active).
            active BOOLEAN,
            -- Whether the market has closed (API: closed).
            closed BOOLEAN,
            -- When the market was created (API: createdAt).
            created_at TIMESTAMP WITH TIME ZONE,
            -- When the market was last updated (API: updatedAt).
            updated_at TIMESTAMP WITH TIME ZONE
        )
    """)


def get_latest_trade_timestamp(conn: duckdb.DuckDBPyConnection, user: str) -> int | None:
    row = conn.execute(
        "SELECT epoch(MAX(timestamp))::BIGINT FROM trades WHERE proxy_wallet = ?", [user]
    ).fetchone()[0]
    return int(row) if row is not None else None


def load_trades(conn: duckdb.DuckDBPyConnection, user: str) -> int:
    """Fetch trades for user (optionally since unix timestamp) and insert into the trades table. Returns (count, trades_df)."""
    since = get_latest_trade_timestamp(conn, user)
    df = get_trades(user, since=since)

    n = len(df)
    if n == 0:
        since_str = (
            datetime.fromtimestamp(since, timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            if since is not None
            else "None"
        )
        logger.info("No new trades since: %s (user: %s)", since_str, user)
        return 0

    conn.register("_trades_input", df)
    conn.execute("""
        INSERT INTO trades (
            timestamp, condition_id, slug, asset, proxy_wallet, side, size, price, outcome, outcome_index
        )
        SELECT
            timestamp,
            conditionId,
            slug,
            asset,
            proxyWallet,
            side,
            size,
            price,
            outcome,
            outcomeIndex
        FROM _trades_input
    """)
    conn.unregister("_trades_input")

    logger.info("Loaded %d trades into database (user: %s)", n, user)
    return n


def load_markets(conn: duckdb.DuckDBPyConnection, condition_ids: list[str]) -> int:
    """Fetch markets by condition_ids and upsert into the markets table."""
    if not condition_ids:
        return 0

    closed = set(
        row[0]
        for row in conn.execute(
            "SELECT condition_id FROM markets WHERE closed = true"
        ).fetchall()
    )
    to_fetch = [c for c in condition_ids if c not in closed]
    if not to_fetch:
        logger.info("All condition_ids are already closed in the database; skipping market fetch.")
        return 0

    df = get_markets(to_fetch)
    if df.empty:
        return 0

    df["categories"] = df["categories"].apply(
        lambda cats: json.dumps(
            [c["label"] for c in cats] if isinstance(cats, list) else []
        )
    )

    conn.register("_markets_input", df)
    conn.execute("""
        INSERT INTO markets (
            id, condition_id, question, slug, description, outcomes, outcome_prices,
            start_date, end_date, volume_num, active, closed, created_at, updated_at
        )
        SELECT
            id,
            "conditionId" AS condition_id,
            question,
            slug,
            description,
            outcomes,
            "outcomePrices" AS outcome_prices,
            categories,
            "startDate" AS start_date,
            "endDate" AS end_date,
            "volumeNum" AS volume_num,
            active,
            closed,
            "createdAt" AS created_at,
            "updatedAt" AS updated_at
        FROM _markets_input
        ON CONFLICT (id) DO UPDATE SET
            condition_id = excluded.condition_id,
            question = excluded.question,
            slug = excluded.slug,
            description = excluded.description,
            outcomes = excluded.outcomes,
            outcome_prices = excluded.outcome_prices,
            categories = excluded.categories,
            start_date = excluded.start_date,
            end_date = excluded.end_date,
            volume_num = excluded.volume_num,
            active = excluded.active,
            closed = excluded.closed,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at
    """)
    n = len(df)
    conn.unregister("_markets_input")
    logger.info("Loaded %d markets into database", n)
    return n
