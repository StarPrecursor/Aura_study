import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from artool import ar_io

# disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Settings
date_start = datetime.datetime(2022, 10, 19)
date_end = datetime.datetime(2022, 11, 8)
frp = ar_io.processors.FundingRateProcessor(date_start, date_end).get_symbol_list(logic="and")
features = ["time", "lastFundingRate", "interestRate", "amount"]

# Paths
df_dir = Path("/home/yangzhe/data/binance/data/futures/um/daily/klines")
br_dir = Path("/home/shared/fundingfee")

# Get borrow data
df_list = []
for br_path in br_dir.glob("*.csv"):
    df_tmp = pd.read_csv(br_path)
    df_list.append(df_tmp)
df_br = pd.concat(df_list, axis=0)
df_br.reset_index(drop=True, inplace=True)

def generate_symbol(df, symbol):
    df_out = df.copy()
    df_out["time_H"] = pd.to_datetime(df_out["time"], unit="ms").dt.floor("H")
    # aggreate
    agg_dict = {
        "lastFundingRate": ["last", "min", "max"],
        "interestRate": ["last"],
        "amount": ["mean"],
    }
    df_agg = df_out.groupby("time_H").agg(agg_dict)
    df_agg = df_agg.shift(1)
    df_agg.columns = ["__".join(x) for x in df_agg.columns.ravel()]
    df_agg.reset_index(inplace=True)
    # rename
    rename_dict = {
        "interestRate__last": "interest",
        "amount__mean": "amount",
    }
    df_agg.rename(columns=rename_dict, inplace=True)
    # save
    df_path_out = df_dir / symbol / "1m" / "features_br.feather"
    df_agg.to_feather(df_path_out)
    return 0

for symbol in ["BTCUSDT"]:  # for debugging
    print(f"# Processing {symbol}")
    try:
        df_syb = df_br.loc[df_br["symbol"] == symbol, features]
    except:
        print(f">>> Failed to load {symbol}")
        continue
    generate_symbol(df_syb, symbol)
    print(f"{symbol} done")
