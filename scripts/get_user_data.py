import argparse
from src.fetch import get_closed_positions

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket user closed positions and save to Parquet."
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

    df = get_closed_positions(args.user)
    columns_to_print = [
        "avgPrice",
        "totalBought",
        "realizedPnl",
        "eventSlug",
        "outcome",
        "endDate",
        "timestamp"
    ]
    print(df[columns_to_print])

    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()
