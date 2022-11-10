import datetime
import warnings
from asyncio import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from artool import ar_io

# disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# get symbols
date_start = datetime.datetime(2022, 1, 1)
#date_end = datetime.datetime(2022, 9, 1)
date_end = datetime.datetime(2022, 11, 8)
frp = ar_io.processors.FundingRateProcessor(date_start, date_end)
#symbols = frp.get_symbol_list(logic="and")
symbols = frp.get_symbol_list(logic="or")

#replace = False
replace = True

def generate_symbol(symbol, trading_type="um"):
    df_dir = Path("/home/yangzhe/data/binance/data/futures/um/daily/klines")
    if trading_type == "spot":
        df_dir = Path("/home/yangzhe/data/binance/data/spot/daily/klines")

    # Check if available
    save_dir = df_dir / symbol / "1m"
    if not save_dir.exists():
        print(f"Folder {save_dir} does not exist. Skip.")
        return 0
    df_path_out = df_dir / symbol / "1m" / "features_H.feather"

    # Check if need rerun
    if df_path_out.exists() and not replace:
        print(f"File {df_path_out} exists. Skip.")
        return 0
    
    # Inputs
    df_path = df_dir / symbol / "1m" / "merge.feather"
    if not df_path.exists():
        return -1

    df = pd.read_feather(df_path)
    df["time"] = pd.to_datetime(df["Open time"], unit="ms")
    df_out = pd.DataFrame()
    df_out["time"] = df["time"]
    df_out["open"] = df["Open"]
    df_out["high"] = df["High"]
    df_out["low"] = df["Low"]
    df_out["close"] = df["Close"]
    df_out["vol"] = df["Volume"]
    df_out["vol_taker_buy"] = df["Taker buy base asset volume"]
    df_out["vol_taker_sell"] = df_out["vol"] - df_out["vol_taker_buy"]
    df_out["vol_taker_buy_ratio"] = df_out["vol_taker_buy"] / df_out["vol"]
    df_out["vol_taker_sell_ratio"] = df_out["vol_taker_sell"] / df_out["vol"]
    # sort by time
    df_out = df_out.sort_values(by="time")

    # Aggreagte to 1H
    ##print("Aggreagte to 1H")
    agg_dict = {
            "high": "max",
            "low": "min",
            "close": ["last", "mean", "std", "median", "skew"],
            "vol": ["sum", "mean", "std", "median", "skew"],
            "vol_taker_buy": "sum",
            "vol_taker_sell": "sum",
            "vol_taker_buy_ratio": ["last", "mean", "std", "median", "skew"],
            "vol_taker_sell_ratio": ["last", "mean", "std", "median", "skew"],
        }
    # get time accurate to hour
    df_out["time_H"] = df_out["time"].dt.floor("H")
    # EWM features
    for hl in [5, 30, 60]:  # in minutes
        df_out[f"price_ewm_{hl}m"] = df_out["close"].ewm(span=hl).mean()
        agg_dict[f"price_ewm_{hl}m"] = "last"
    for hl in [4, 8, 24]:  # in hours
        hl_min = hl * 60
        df_out[f"price_ewm_{hl}h"] = df_out["close"].ewm(span=hl_min).mean()
        agg_dict[f"price_ewm_{hl}h"] = "last"
    for hl in [2, 7, 14]:  # in days
        hl_min = hl * 24 * 60
        df_out[f"price_ewm_{hl}d"] = df_out["close"].ewm(span=hl_min).mean()
        agg_dict[f"price_ewm_{hl}d"] = "last"
    # group by hour
    df_out_agg = df_out.groupby("time_H").agg(agg_dict)
    # shift by 1 (don't use data at current Hour)
    df_out_agg = df_out_agg.shift(1)
    # flatten multi-index
    df_out_agg.columns = ["__".join(x) for x in df_out_agg.columns.ravel()]
    df_out_agg.reset_index(inplace=True)

    # Post process
    # convert float64 to float32
    #for col in df_out_agg.columns:
    #    if df_out_agg[col].dtype == "float64":
    #        df_out_agg[col] = df_out_agg[col].astype("float32")

    # modify column names
    rename_dict = {"high__max": "price__high", "low__min": "price__min", "vol__sum":"vol"}
    df_out_agg.rename(columns=rename_dict, inplace=True)
    for col in df_out_agg.columns:
        if col.startswith("close__"):
            df_out_agg.rename(columns={col: col.replace("close__", "price__")}, inplace=True)
    for col in df_out_agg.columns:
        if col.endswith("__last"):
            df_out_agg.rename(columns={col: col.replace("__last", "")}, inplace=True)

    # Post features
    # price_ema diff with price
    ema_features = [x for x in df_out_agg.columns if x.startswith("price_ema")]
    df_new_cols = ar_io.ar_fe.get_diff_features(df_out_agg, ema_features, "price")
    df_out_agg = pd.concat([df_out_agg, df_new_cols], axis=1)

    # Save to feather
    df_out_agg.to_feather(df_path_out)
    return 0

with ProcessPoolExecutor() as executor:
    futures = []
    #symbols = ["BTCUSDT"]  # for debugging
    for trading_type in ["um", "spot"]:
        for symbol in symbols:
            futures.append(executor.submit(generate_symbol, symbol, trading_type))
    for i, future in tqdm(enumerate(futures)):
        symbol_id = i % len(symbols)
        type_id = i // len(symbols)
        symbol = symbols[symbol_id]
        trading_type = ["um", "spot"][type_id]
        if future.result() == -1:
            print(f"{symbol} ({trading_type}) don't have inputs, skipped")
        else:
            print(f"{symbol} ({trading_type}) done")
