import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from artool import ar_ana, ar_model, ar_io

# set logging level
logging.basicConfig(level=logging.INFO)
logging.getLogger("artool").setLevel(logging.INFO)

# Trade parameters
cap = 1e6
fut = 10
fee = 3e-4
vol_lim = 0.05
hold_lim = 0.2

# Config
model_config_path = (
    Path("/home/yangzhe/Aura_study/model/fr_pred/1030_fut_n_trade")
    / f"config_fut_{fut}.yaml"
)
with open(model_config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# Get inputs
print("Loading inputs...")
input_dir = Path(cfg["input_dir"])
df = pd.read_feather(input_dir / "input.feather")
split_time = pd.to_datetime(cfg["train_test_split_date"])
#split_time = pd.to_datetime("2022-05-01")

# Predict
print("Predicting...")
model_dir = Path(cfg["model_dir"])
model = ar_model.fr_model.FRModel_LGB(cfg)
model.load_model(model_dir, cfg["model_name"])
pred = model.predict(df[cfg["features"]])
df["signal"] = pred


# Study
df_tr, _, df_te = ar_model.train_utils.data_split_by_time(
    df, cfg["time"], split_time, val_ratio=0
)
symbols = df_te["symbol"].unique()

# Optimize buy/sell parameters
opt_buy = 0
opt_sell = 0
for data_cat, df_cur in [("train", df_tr), ("test", df_te)]:
    plot_dir = Path(f"scan/{data_cat}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Prepare trade data
    trade_data = ar_io.io_utils.decouple_df(
        df_cur,
        on="symbol",
        within=symbols,
        feature_map={
            "time": cfg["time"],
            "price": "price",
            "funding_rate": "funding_rate",
            "vol": "vol",
            "signal": "signal",
            "signal_true": "funding_rate_trade__future_10",
        },
    )

    ts = ar_ana.simu.TradeSimulatorSignalSimple(trade_data)
    ts.fee = fee
    ts.vol_lim = 0.05
    ts.hold_lim = 0.2

    def get_score(buy, sell, ts=ts, cap=1e6):
        ts_cur = copy.copy(ts)
        ts_cur.set_buy_point(buy)
        ts_cur.set_sell_point(sell)
        ts_cur.trade(cap, show_progress=False)
        # return ts_cur.get_total_pnl()
        profit_rate_per_year = ts_cur.get_total_pnl() / cap
        if data_cat == "train":
            return profit_rate_per_year * 12 / 5
        elif data_cat == "test":
            return profit_rate_per_year * 12 / 3

    # scan
    scan_dim = {
        "buy": np.linspace(-1e-4, 1e-4, 21),
        "sell": np.linspace(-2e-4, 0, 21),
    }
    df_scan = ar_ana.scan.grid_scan(scan_dim, get_score)

    # find sell/buy point of highest score
    df_top = df_scan.sort_values("score", ascending=False)
    df_top = df_top.head(1)
    best_buy = df_top["buy"].values[0]
    best_sell = df_top["sell"].values[0]
    best_score = df_top["score"].values[0]
    print(f"Optimal buy/sell point: {best_buy}, {best_sell}")

    if data_cat == "train":
        opt_buy = best_buy
        opt_sell = best_sell
    else:
        ts_cur = copy.copy(ts)
        ts_cur.set_buy_point(opt_buy)
        ts_cur.set_sell_point(opt_sell)
        ts_cur.trade(cap, show_progress=False)
        opt_score = ts_cur.get_total_pnl() / cap * 12 / 3
        print(f">>>> Optimal score: {opt_score}")
        test_train_diff = (opt_score - best_score) / best_score

    # plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    df_tmp = df_scan.round(8).pivot("sell", "buy", "score")
    df_tmp[df_tmp < 0] = 0  # remove negative values
    sns.heatmap(df_tmp, ax=ax)
    fig.savefig(plot_dir / f"trade_scan_{data_cat}.png")

    # Single simulation
    ts.set_buy_point(opt_buy)
    ts.set_sell_point(opt_sell)
    ts.trade(cap)
    ts.plot_book(plot_dir)
    # return ts_cur.get_total_pnl()
    profit_rate_per_year = ts.get_total_pnl() / cap
    if data_cat == "train":
        print(f"Train profit rate per year: {profit_rate_per_year * 12 / 5}")
    elif data_cat == "test":
        print(f"Test profit rate per year: {profit_rate_per_year * 12 / 3}")
