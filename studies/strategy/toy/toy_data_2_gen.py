import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from scipy.signal import lfilter

from artool import ar_io, toy


def process_df(df_in, features_save, features_fe, save_path):
    df = df_in[features_save].copy()
    new_cols = {}
    for feature in features_fe:
        # Add exponential decay sum
        for lookback in [0.5, 1, 2, 3, 5, 10, 20, 50]:
            decay_rate = np.exp(-1 / lookback)
            new_cols[f"{feature}_expcumsum_{lookback}"] = lfilter(
                [decay_rate], [1, -decay_rate], df[feature]
            )
        # Add rolling features
        for lookback in [3, 5, 10, 20, 50]:
            new_cols[f"{feature}_rol_mean_{lookback}"] = (
                df[feature].rolling(lookback, min_periods=1).mean()
            )
            new_cols[f"{feature}_rol_std_{lookback}"] = (
                df[feature].rolling(lookback, min_periods=1).std()
            )
            new_cols[f"{feature}_rol_max_{lookback}"] = (
                df[feature].rolling(lookback, min_periods=1).max()
            )
            new_cols[f"{feature}_rol_min_{lookback}"] = (
                df[feature].rolling(lookback, min_periods=1).min()
            )
            new_cols[f"{feature}_rol_skew_{lookback}"] = (
                df[feature].rolling(lookback, min_periods=1).skew()
            )
            if lookback > 3:
                new_cols[f"{feature}_rol_kurt_{lookback}"] = (
                    df[feature].rolling(lookback, min_periods=1).kurt()
                )
        # Add difference bewteen cumsum with 5 interval
        for lookforward in [1, 3, 5, 10]:
            new_cols[f"funding_rate_future_{lookforward}"] = (
                df["funding_rate"].rolling(lookforward).sum().shift(-lookforward)
            )

    df_append = pd.DataFrame(new_cols)
    df_out = pd.concat([df, df_append], axis=1)
    df_out.to_feather(save_path)
    return f"Saved {save_path}"


features = ["funding_timestamp", "funding_rate", "index_price", "mark_price"]
date_start = datetime.datetime(2022, 1, 1)
date_end = datetime.datetime(2022, 9, 1)
symbols = toy.toy_data.get_symbol_list(date_start, date_end)

# get_symbol_data_coarse
print("get_symbol_data_coarse")
df_list = []
for symbol in tqdm.tqdm(symbols):
    df = toy.toy_data.get_symbol_data_coarse(symbol, date_start, date_end, features)
    df_list.append(df)

# process
print("processing")
features_save = [
    "symbol",
    "funding_timestamp",
    "funding_rate",
    "funding_rate",
    "open_interest",
    "last_price",
    "index_price",
    "mark_price",
]
features_fe = ["funding_rate", "index_price", "mark_price"]

save_dir = Path("/home/yangzhe/data/toy_data_2")
save_dir.mkdir(parents=True, exist_ok=True)

with ProcessPoolExecutor(max_workers=24) as executor:
    futures = []
    for df in df_list:
        symbol = df["symbol"].iloc[0]
        save_path = save_dir / f"{symbol}.feather"
        futures.append(
            executor.submit(process_df, df, features_save, features_fe, save_path)
        )
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        future.result()
