import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from artool import ar_ana, ar_model

# set logging level
logging.basicConfig(level=logging.INFO)
logging.getLogger("artool").setLevel(logging.INFO)


# Config
model_config_path = Path("/home/yangzhe/Aura_study/model/fr_pred/1028_fut_n/config_fut_10.yaml")
with open(model_config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Get inputs
print("Loading inputs...")
input_dir = Path(cfg["input_dir"])
df = pd.read_feather(input_dir / "input.feather")
split_time = pd.to_datetime(cfg["train_test_split_date"])

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
for data_cat, df_cur in [("train", df_tr), ("test", df_te)]:
    plot_dir = Path(f"plot_{data_cat}")
    plot_dir.mkdir(exist_ok=True)

    # Plot y_te_pred hist
    fig, ax = plt.subplots()
    hist, edges = np.histogram(df_cur["signal"], bins=100)
    ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge")
    ax.set_title(f"{data_cat} signal hist")
    ax.set_xlabel("signal")
    ax.set_ylabel("count")
    fig.savefig(plot_dir / f"{data_cat}_signal.png")

    # Simu trade
    trade_data = {}
    for symbol in symbols:
        df_tmp = df_cur[df_cur["symbol"] == symbol]
        df_syb = pd.DataFrame()
        df_syb["time"] = df_tmp[cfg["time"]]
        df_syb["price"] = df_tmp["price"]
        df_syb["funding_rate"] = df_tmp["funding_rate"]
        df_syb["vol"] = df_tmp["vol"]
        df_syb["signal"] = df_tmp["signal"]
        trade_data[symbol] = df_syb
    ts = ar_ana.simu.TradeSimulatorSignalSimple(trade_data)

    def get_score(buy, sell, ts=ts, cap=1e6):
        ts_cur = copy.copy(ts)
        ts_cur.set_buy_point(buy)
        ts_cur.set_sell_point(sell)
        ts_cur.vol_lim = 0.01
        ts_cur.hold_lim = 0.2
        ts_cur.trade(cap, show_progress=False)
        #return ts_cur.get_total_pnl()
        profit_rate_per_year = ts_cur.get_total_pnl() / cap
        if data_cat == "train":
            return profit_rate_per_year * 12 / 5
        elif data_cat == "test":
            return profit_rate_per_year * 12 / 3

    # scan
    scan_dim = {
        "buy": np.linspace(0, 1e-4, 11),
        "sell": np.linspace(-1e-4, 0, 11),
    }
    df_scan = ar_ana.scan.grid_scan(scan_dim, get_score)

    # find sell/buy point of highest score
    df_top = df_scan.sort_values("score", ascending=False)
    df_top = df_top.head(1)
    best_buy = df_top["buy"].values[0]
    best_sell = df_top["sell"].values[0]
    print(f"Optimal buy/sell point: {best_buy}, {best_sell}")

    # plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    df_tmp = df_scan.round(8).pivot("sell", "buy", "score")
    df_tmp[df_tmp < 0] = 0  # remove negative values
    sns.heatmap(df_tmp, ax=ax)
    fig.savefig(plot_dir / f"trade_scan_{data_cat}.png")
