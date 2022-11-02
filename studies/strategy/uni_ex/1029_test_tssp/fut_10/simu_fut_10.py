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
    plot_dir = Path(f"simu/{data_cat}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Plot y_te_pred hist
    fig, ax = plt.subplots()
    hist, edges = np.histogram(df_cur["signal"], bins=100)
    ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge")
    ax.set_title(f"{data_cat} signal hist")
    ax.set_xlabel("signal")
    ax.set_ylabel("count")
    fig.savefig(plot_dir / f"signal_{data_cat}.png")

    # Simu trade
    cap = 1e6
    buy_point = 3e-5

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
    tssp = ar_ana.simu.TradeSimulatorSignalPeriodic(trade_data, period=20)
    tssp.set_buy_point(buy_point)
    tssp.vol_lim = 0.01
    tssp.hold_lim = 0.2
    tssp.trade(cap)
    tssp.plot_book(plot_dir)
    #return ts_cur.get_total_pnl()
    profit_rate_per_year = tssp.get_total_pnl() / cap
    if data_cat == "train":
        print(f"Train profit rate per year: {profit_rate_per_year * 12 / 5}")
    elif data_cat == "test":
        print(f"Test profit rate per year: {profit_rate_per_year * 12 / 3}")
