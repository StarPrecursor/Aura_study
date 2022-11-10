import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from artool import ar_io

# disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

df_dir = Path("/home/yangzhe/data/binance/data/futures/um/daily/klines")

date_start = datetime.datetime(2022, 1, 1)
#date_end = datetime.datetime(2022, 9, 1)
date_end = datetime.datetime(2022, 11, 8)
frp = ar_io.processors.FundingRateProcessor(date_start, date_end)
symbols = frp.get_symbol_list(logic="and")
features = [
    "timestamp",
    "funding_timestamp",
    "funding_rate",
    "index_price",
    "mark_price",
]


def generate_symbol(df_fr, symbol):
    # Add rolling features
    df_new_cols = ar_io.ar_fe.get_rolling_features(
        df_fr, ["funding_rate"], [3, 5, 10, 20, 50]
    )
    df_fr = pd.concat([df_fr, df_new_cols], axis=1)

    # EWM features
    agg_cols = []
    # min
    df_new_cols = ar_io.ar_fe.get_ewm_features(
        df_fr, ["funding_rate"], [5, 30, 60], suffix="m"
    )
    df_fr = pd.concat([df_fr, df_new_cols], axis=1)
    agg_cols += df_new_cols.columns.tolist()
    df_new_cols = ar_io.ar_fe.get_delta_features(df_fr, df_new_cols.columns)
    df_fr = pd.concat([df_fr, df_new_cols], axis=1)
    agg_cols += df_new_cols.columns.tolist()
    # hour
    df_new_cols = ar_io.ar_fe.get_ewm_features(
        df_fr, ["funding_rate"], [4, 8, 24], scale=60, suffix="h"
    )
    df_fr = pd.concat([df_fr, df_new_cols], axis=1)
    agg_cols += df_new_cols.columns.tolist()
    df_new_cols = ar_io.ar_fe.get_delta_features(df_fr, df_new_cols.columns)
    df_fr = pd.concat([df_fr, df_new_cols], axis=1)
    agg_cols += df_new_cols.columns.tolist()
    # day
    df_new_cols = ar_io.ar_fe.get_ewm_features(
        df_fr, ["funding_rate"], [2, 7, 14], scale=60 * 24, suffix="d"
    )
    df_fr = pd.concat([df_fr, df_new_cols], axis=1)
    agg_cols += df_new_cols.columns.tolist()
    df_new_cols = ar_io.ar_fe.get_delta_features(df_fr, df_new_cols.columns)
    df_fr = pd.concat([df_fr, df_new_cols], axis=1)
    agg_cols += df_new_cols.columns.tolist()

    # Aggregate to hours
    agg_dict = {
        "funding_rate": ["mean", "std", "max", "min", "skew", "last"],
    }
    for ky in df_new_cols.columns:
        agg_dict[ky] = ["last"]
    df_fr["time_H"] = df_fr["time"].dt.floor("H")
    for ky in agg_cols:
        agg_dict[ky] = ["last"]
    df_fr_agg = df_fr.groupby("time_H").agg(agg_dict)  # type: ignore
    del df_fr

    # Use previous past one hour's data
    df_fr_agg = df_fr_agg.shift(1)
    df_fr_agg.columns = ["__".join(x) for x in df_fr_agg.columns.ravel()]  # type: ignore
    df_fr_agg.reset_index(inplace=True)

    # rename columns
    for col in df_fr_agg.columns:
        if col.endswith("__last"):
            df_fr_agg.rename(columns={col: col.replace("__last", "")}, inplace=True)

    # Add future funding rate mean
    df_new_cols = ar_io.ar_fe.get_future_features(
        df_fr_agg, ["funding_rate"], [1, 3, 5, 10, 15, 21, 30, 45], scale=8
    )
    df_fr_agg = pd.concat([df_fr_agg, df_new_cols], axis=1)
    # Add future funding rate mean at trade hour 0/8/16
    trade_idx = df_fr_agg["time_H"].dt.hour.isin([0, 8, 16])
    df_fr_agg["funding_rate_trade"] = df_fr_agg["funding_rate"]
    df_fr_agg.loc[~trade_idx, "funding_rate_trade"] = np.nan
    df_new_cols = ar_io.ar_fe.get_future_features(
        df_fr_agg,
        ["funding_rate_trade"],
        [1, 3, 5, 10, 15, 21, 30, 45],
        scale=8,
        min_periods=1,
    )
    df_fr_agg = pd.concat([df_fr_agg, df_new_cols], axis=1)

    # Save
    save_path = df_dir / symbol / "1m" / "features_fr.feather"
    df_fr_agg.to_feather(save_path)
    return 0


for symbol in tqdm(symbols):
# for symbol in ["BTCUSDT"]:  # for debugging
    print(f"# Processing {symbol}")
    try:
        df_fr = frp.get_symbol_data(symbol, features, use_8h=False, resample="1min")
        df_fr["time"] = pd.to_datetime(df_fr["timestamp"], unit="us")
        df_fr = df_fr.sort_values(by="time")
    except:
        print(f">>> Failed to load {symbol}")
        continue
    generate_symbol(df_fr, symbol)
    print(f"{symbol} done")
