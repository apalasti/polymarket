"""Fetch Polymarket data from the Data API and Dome API."""

import logging
import os
from datetime import datetime, timezone

from dome_api_sdk.types import OrderbookSnapshot
import pandas as pd
import requests

from dome_api_sdk import DomeClient


logger = logging.getLogger(__name__)
BASE_URL = "https://data-api.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# Dome API client (requires DOME_API_KEY env var for Dome calls)
_dome_client: DomeClient | None = None


def _get_dome_client() -> DomeClient:
    """Return Dome client; requires DOME_API_KEY to be set."""
    global _dome_client
    if _dome_client is None:
        api_key = os.environ.get("DOME_API_KEY")
        if not api_key:
            raise ValueError("DOME_API_KEY environment variable is required for Dome API calls")
        _dome_client = DomeClient({"api_key": api_key, "timeout": 30})
    return _dome_client


def get_closed_positions(
    user: str,
    since: int | None = None,
) -> pd.DataFrame:
    """Fetch all closed positions for a Polymarket user as a DataFrame.

    Paginates through the API (max 50 per request) until all positions are retrieved.

    Args:
        user: Ethereum address of the user (e.g. 0x56687bf447db6ffa42ffe2204a05edaa20f55839).
        since: Optional unix timestamp; only return data up to this time.

    Returns:
        DataFrame of all closed positions.
    """
    params: dict = {
        "user": user,
        "offset": 0,
        "limit": 50,
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC",
    }

    rows: list[dict] = []
    while True:
        resp = requests.get(
            f"{BASE_URL}/closed-positions",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "Fetched page %d (offset=%d): %d positions (user: %s)",
            int(params["offset"] / params["limit"] + 1),
            params["offset"],
            len(data),
            user,
        )

        if since is not None:
            data = [r for r in data if since < r["timestamp"]]

        rows.extend(data)
        if len(data) < params["limit"]:
            break
        params["offset"] += params["limit"]

    return pd.DataFrame(rows)


def get_trades(
    user: str,
    since: int | None = None,
) -> pd.DataFrame:
    """Fetch all trades for a Polymarket user as a DataFrame.

    Paginates through the API (max 10_000 per request) until all trades are retrieved.

    Args:
        user: Ethereum address of the user (e.g. 0x56687bf447db6ffa42ffe2204a05edaa20f55839).
        taker_only: If True, only return trades where the user was the taker. (Executed immidiately.)
        since: Optional unix timestamp; only return data after this time.

    Returns:
        DataFrame of all trades.
    """
    params: dict = {
        "user": user,
        "offset": 0,
        "limit": 10_000,
        "takerOnly": str(False).lower(),
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC",
    }

    rows: list[dict] = []
    while True:
        resp = requests.get(
            f"{BASE_URL}/trades",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data: list[dict] = resp.json()
        logger.info(
            "Fetched page %d (offset=%d): %d trades (user: %s)",
            int(params["offset"] / params["limit"] + 1),
            params["offset"],
            len(data),
            user,
        )

        if since is not None:
            data = [r for r in data if since < r["timestamp"]]

        rows.extend(data)
        if len(data) < params["limit"]:
            break
        params["offset"] += params["limit"]

    df = pd.DataFrame(rows)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def get_markets(condition_ids: list[str]) -> pd.DataFrame:
    """Fetch markets from the Gamma API by condition IDs.

    Args:
        condition_ids: List of market condition IDs.

    Returns:
        List of market objects from the API.
    """
    if not condition_ids:
        return []
    resp = requests.get(
        f"{GAMMA_URL}/markets",
        params={"condition_ids": condition_ids},
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    return df


def get_markets_by_slug(slugs: list[str]) -> list[dict]:
    """Fetch markets from the Gamma API by slug(s).

    Args:
        slugs: List of market slugs (e.g. ["btc-updown-15m-1762516800"]).

    Returns:
        List of market objects. Markets not found are omitted.
    """
    if not slugs:
        return []
    resp = requests.get(
        f"{GAMMA_URL}/markets",
        params=[("slug", s) for s in slugs],
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _to_utc_ts(t: datetime | int | float) -> int:
    """Convert to Unix timestamp in seconds (naive datetime treated as UTC)."""
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return int(t.timestamp())
    if isinstance(t, (int, float)) and t >= 1e12:
        return int(t) // 1000  # ms -> seconds
    return int(t)  # already seconds


def get_orderbook_history(
    token_id: str,
    start_time: datetime | int | float,
    end_time: datetime | int | float,
    limit: int = 200,
):
    """Fetch full orderbook history, paginating until no more snapshots."""
    client = _get_dome_client()
    params: dict = {
        "token_id": token_id,
        "start_time": _to_utc_ts(start_time) * 1000,
        "end_time": _to_utc_ts(end_time) * 1000,
        "limit": limit,
    }

    all_snapshots: list[OrderbookSnapshot] = []
    httpx_logger = logging.getLogger("httpx")
    saved_level = httpx_logger.level
    try:
        httpx_logger.setLevel(logging.WARNING)
        while True:
            result = client.polymarket.markets.get_orderbooks(params)
            all_snapshots.extend(result.snapshots)
            if not result.pagination.has_more or result.pagination.pagination_key is None:
                break
            params["pagination_key"] = result.pagination.pagination_key
    finally:
        httpx_logger.setLevel(saved_level)
    return all_snapshots
