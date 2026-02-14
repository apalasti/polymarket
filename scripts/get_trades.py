import argparse

import pandas as pd

from src.fetch import get_trades


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket user trades and save to Parquet."
    )
    parser.add_argument(
        "--user", "-u",
        required=True,
        help="Ethereum address of the user."
    )
    parser.add_argument(
        "--output", "-o",
        required=False,
        help="Output Parquet filename (default: <user>.parquet)."
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{args.user}.parquet"

    cols = [
        "slug", "conditionId", "asset", "proxyWallet", "side",
        "size", "price", "outcome", "outcomeIndex", "timestamp"
    ]
    df = get_trades(args.user)[cols]

    with pd.option_context(
        "display.expand_frame_repr",
        False,
        "display.max_columns",
        None,
        "display.max_colwidth",
        20,
    ):
        print("Most recent 50 trades:")
        print(df.head(50))

    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
