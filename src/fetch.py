"""Fetch Polymarket data from the Data API."""

import pandas as pd
import requests

BASE_URL = "https://data-api.polymarket.com"


def get_closed_positions(
    user: str,
    *,
    page_size: int = 50,
    sort_by: str = "timestamp",
    sort_direction: str = "ASC",
) -> pd.DataFrame:
    """Fetch all closed positions for a Polymarket user as a DataFrame.

    Paginates through the API (max 50 per request) until all positions are retrieved.

    Args:
        user: Ethereum address of the user (e.g. 0x56687bf447db6ffa42ffe2204a05edaa20f55839).
        limit: Page size per request (max 50).
        sort_by: Sort field (e.g. timestamp, realizedPnl, title, price, avgPrice).
        sort_direction: ASC or DESC.

    Returns:
        DataFrame of all closed positions.
    """
    offset = 0
    page_size = min(page_size, 50)

    rows: list[dict] = []
    while True:
        resp = requests.get(
            f"{BASE_URL}/closed-positions",
            params={
                "user": user,
                "limit": page_size,
                "offset": offset,
                "sortBy": sort_by,
                "sortDirection": sort_direction,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        if len(data) < page_size:
            break
        offset += page_size

    return pd.DataFrame(rows)
