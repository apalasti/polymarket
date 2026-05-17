import pandas as pd
import numpy as np

def filter_and_downsample(filepath: str) -> pd.DataFrame:
    """
    Stable data processing steps that shouldn't change between feature engineering iterations.
    - Filters to BTC markets only.
    - Defines the binary target variable.
    - Downsamples to 1-minute intervals (last known state).
    """
    print("Loading data for base processing...")
    df = pd.read_parquet(filepath)
    
    # 1. Filter to BTC markets only
    print("Filtering for BTC markets...")
    df = df[df['slug'].str.contains('btc-updown', case=False, na=False)].copy()
    
    # 2. Target definition
    print("Mapping target...")
    df['target'] = df['outcome_price']
    df = df.dropna(subset=['target'])
    
    # 3. Sub-sample to 1-minute intervals (taking the last known state per minute)
    print("Sub-sampling to 1-minute intervals...")
    df['minute'] = df['seconds'] // 60
    df = df.sort_values(['slug', 'seconds']).groupby(['slug', 'minute']).tail(1).copy()
    
    return df
