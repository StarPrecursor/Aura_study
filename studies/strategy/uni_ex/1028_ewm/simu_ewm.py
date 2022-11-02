import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from sklearn.metrics import r2_score

from artool import ar_ana, ar_model

# set logging level
logging.basicConfig(level=logging.INFO)
logging.getLogger("artool").setLevel(logging.INFO)


# Config
input_dir = Path("/home/yangzhe/model/fr_pred/1027_lgb")

model_config_path = Path("/home/yangzhe/Aura_study/model/fr_pred/1027_lgb/config.yaml")
with open(model_config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Get inputs
print("Loading inputs...")
df = pd.read_feather(input_dir / "input.feather")
split_time = pd.to_datetime(cfg["train_test_split_date"])
df_tr, _, df_te = ar_model.train_utils.data_split_by_time(
    df, cfg["time"], split_time, val_ratio=0
)
symbols = df_te["symbol"].unique()

for data_cat, df_cur in [("train", df_tr), ("test", df_te)]:
    y_sig = df_cur["funding_rate__ewm2d"]
    df_cur = df_cur.assign(signal=y_sig)
    plot_dir = Path(f"plot_{data_cat}")
    plot_dir.mkdir(exist_ok=True)

    # plot funding_rate__ewm2d vs funding_rate__future_
    for n_fut in [1, 3, 5, 10]:
        fig, ax = plt.subplots()
        x = df_cur["funding_rate__ewm2d"]
        y = df_cur[f"funding_rate__future_{n_fut}"]
        idx_nan = np.isnan(x) | np.isnan(y)
        x = x[~idx_nan]
        y = y[~idx_nan]
        r2 = r2_score(y, x)
        corr = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=1, alpha=0.1)
        ax.set_title(f"r2={r2:.3f}, corr={corr:.3f}")
        ax.set_xlabel("funding_rate__ewm2d")
        ax.set_ylabel(f"funding_rate__future_{n_fut}")
        fig.savefig(plot_dir / f"fr_ewm2d_vs_fr_fut_{n_fut}.png")

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
    ts.trade(1e6)
    ts.plot_book(plot_dir)
