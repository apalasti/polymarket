import pandas as pd
from typing import Tuple

def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stable splitting logic to prevent data leakage.
    Splits markets chronologically into Train, Val, and Test sets.
    """
    df = df.sort_values('timestamp')
    unique_slugs = df['slug'].unique()
    n_slugs = len(unique_slugs)
    
    train_end = int(n_slugs * train_ratio)
    val_end = int(n_slugs * (train_ratio + val_ratio))
    
    train_slugs = set(unique_slugs[:train_end])
    val_slugs = set(unique_slugs[train_end:val_end])
    test_slugs = set(unique_slugs[val_end:])
    
    train_df = df[df['slug'].isin(train_slugs)].copy()
    val_df = df[df['slug'].isin(val_slugs)].copy()
    test_df = df[df['slug'].isin(test_slugs)].copy()
    
    return train_df, val_df, test_df
