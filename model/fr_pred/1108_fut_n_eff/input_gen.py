import datetime
from pathlib import Path

import pandas as pd
import yaml

import git
from artool import ar_io

# Settings
data_dir = Path("/home/yangzhe/data/binance/data/futures/um/daily/klines")
save_dir = Path("/home/yangzhe")
# get path relative to git repo
git_rel = ar_io.io_utils.get_git_rel_path(Path.cwd())
model_dir = save_dir / git_rel
model_dir.mkdir(parents=True, exist_ok=True)
date_start = datetime.datetime(2022, 1, 1)
date_end = datetime.datetime(2022, 9, 1)
frp = ar_io.processors.FundingRateProcessor(date_start, date_end)
symbols = frp.get_symbol_list(logic="and")
print(f"Number of symbols: {len(symbols)}")

# Prepare inputs
print("Preparing inputs...")


def process_symbol(symbol):
    df_dir = data_dir / symbol / "1m"
    df_1 = pd.read_feather(df_dir / "features_H.feather")
    df_2 = pd.read_feather(df_dir / "features_fr.feather")
    # add lag
    new_cols = ar_io.ar_fe.get_lag_features(df_2, "funding_rate", [1, 3, 5, 8, 24])
    lag_cols = new_cols.columns.tolist()
    df_2 = pd.concat([df_2, new_cols], axis=1)
    # add lag diff
    new_cols = ar_io.ar_fe.get_diff_features(df_2, lag_cols, "funding_rate")
    df_2 = pd.concat([df_2, new_cols], axis=1)
    # drop head/tail 24
    df_2 = df_2.iloc[24:-24]
    # merge
    df_out = pd.merge(df_1, df_2, on=["time_H"])
    df_out["symbol"] = symbol

    # add fr_eff
    df_eff = ar_io.ar_fe.get_future_fr_eff(
        df_out,
        #fr_col="funding_rate",
        fr_col="funding_rate_trade",
        price_col="price",
        lookforwards=[1, 3, 5, 10, 15, 21, 30, 45],
        scale=8,
        min_periods=1,
    )
    return pd.concat([df_out, df_eff], axis=1)


#### debug
#symbol = "BTCUSDT"
#df_debug = process_symbol(symbol)
##print(df_debug.columns)
#df_debug = df_debug.loc[
#   :, ["time_H", "funding_rate_trade", "price", "funding_rate_trade__future_3_eff", "funding_rate_trade__future_21_eff"]
#]
#print(df_debug.head(20))
#exit()
####

df = ar_io.processors.merge_symbols_mp(process_symbol, symbols)
print(df.shape)

# Add time features
new_cols = ar_io.ar_fe.get_dt_features(df, "time_H")
new_cols["time_H__hourof8"] = df["time_H"].dt.hour % 8
df = pd.concat([df, new_cols], axis=1)

# Add fear & greed index
df_fear_greed = pd.read_feather(
    "/home/yangzhe/data/external_data/fear_and_greed_index.feather"
)
df_fear_greed["date"] = pd.to_datetime(df_fear_greed["timestamp"]).dt.date
df_fear_greed["fear_and_greed_value"] = df_fear_greed["fear_and_greed_value"].shift(1)
df_fear_greed["fear_and_greed_class"] = df_fear_greed["fear_and_greed_class"].shift(1)
df["date"] = pd.to_datetime(df["time_H"], unit="us").dt.date
df = df.merge(
    df_fear_greed[["date", "fear_and_greed_value", "fear_and_greed_class"]],
    on="date",
    how="left",
)
save_path = model_dir / "input.feather"
df.to_feather(save_path)
print(df.info())

# Prepare input info
print("Preparing input info...")
columns = df.columns.tolist()
features = []
targets = []
time = []
for col in columns:
    if "future" in col:
        targets.append(col)
    elif col.startswith("time_"):
        time.append(col)
    else:
        features.append(col)
info_dict = {
    "date_start": date_start,
    "date_end": date_end,
    "time": time,
    "features": features,
    "targets": targets,
    "symbols": symbols,
}
with open(model_dir / "input_info.yaml", "w") as f:
    yaml.dump(info_dict, f, default_flow_style=False, sort_keys=False)
