import argparse
import csv
import json
import logging
import pathlib
from datetime import datetime, timedelta, timezone

from tqdm import tqdm

from src.fetch import get_orderbook_history
from src.crypto_market import CryptoMarkets15m


DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

BID_ASK_COLS = (
    [x for i in range(1, 6) for x in (f"bid_{i}_price", f"bid_{i}_size")]
    + [x for i in range(1, 6) for x in (f"ask_{i}_price", f"ask_{i}_size")]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    CSV_PATH = DATA_DIR / "crypto_price_history.csv"
    ASSETS_DEFAULT = ["btc", "eth", "sol"]

    parser = argparse.ArgumentParser(
        description="Collect 15m crypto market price history to CSV via Dome API (candlesticks). Set DOME_API_KEY.",
    )
    parser.add_argument(
        "--since",
        required=True,
        help="Start of collection as ISO date/datetime (e.g. 2024-01-15 or 2024-01-15T12:00). Collection runs from this time up to now.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=ASSETS_DEFAULT,
        help=f"Assets for slug pattern (default: {ASSETS_DEFAULT}).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DATA_DIR / "crypto_price_history.csv",
        help=f"Output CSV path (default: {CSV_PATH}).",
    )
    args = parser.parse_args()

    args.since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    args.end = datetime.now(timezone.utc)
    if args.since >= args.end:
        raise argparse.ArgumentError("--since must be before now.")

    return args


def main():
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["slug", "timestamp", "seconds", "outcome", "outcome_price", "price"]
            + BID_ASK_COLS
        )

        for asset in args.assets:
            crypto_market = CryptoMarkets15m(asset, args.since, args.end)

            for market in (
                t := tqdm(
                    crypto_market.iter_markets(),
                    desc=f"Markets for {crypto_market}",
                    unit="market",
                )
            ):
                slug = market.get("slug") or ""

                time_range = CryptoMarkets15m.slug_to_time_range(slug)
                if time_range is None:
                    logger.warning("Could not derive time range from slug %s", slug)
                    continue

                start, end = time_range
                t.set_postfix_str(f"{slug} - {start}")

                token_ids = json.loads(market.get("clobTokenIds", "[]"))
                for token_id in token_ids[:1]:  # Just do it for a single token id
                    try:
                        prices = crypto_market.get_prices(start, end, interval="1s")
                        orderbook = get_orderbook_history(
                            token_id,
                            start - timedelta(seconds=1),
                            end + timedelta(seconds=1),
                        )
                    except Exception as e:
                        logger.warning("Orderbook history failed for market '%s': %s", slug, e)
                        break

                    if not orderbook:
                        break

                    for i, price in enumerate(prices):
                        snap = min(
                            orderbook,
                            key=lambda s: abs(
                                s.timestamp
                                - (start + timedelta(seconds=i)).timestamp() * 1000
                            ),
                        )
                        bids = sorted(snap.bids, key=lambda x: float(x["price"]), reverse=True)[:5]
                        asks = sorted(snap.asks, key=lambda x: float(x["price"]))[:5]
                        row = [
                            slug,
                            snap.timestamp,
                            i,
                            json.loads(market.get("outcomes", '[""]'))[0],
                            json.loads(market.get("outcomePrices", '[""]'))[0],
                            price,
                        ]
                        for level in range(5):
                            b = bids[level] if level < len(bids) else {}
                            row.extend([b.get("price", ""), b.get("size", "")])
                        for level in range(5):
                            a = asks[level] if level < len(asks) else {}
                            row.extend([a.get("price", ""), a.get("size", "")])
                        writer.writerow(row)

    logger.info("Wrote CSV to %s", args.output)


if __name__ == "__main__":
    main()
