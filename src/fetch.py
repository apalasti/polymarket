"""Fetch Polymarket data from the Data API."""

import logging
import pandas as pd
import requests

logger = logging.getLogger(__name__)
BASE_URL = "https://data-api.polymarket.com"


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
