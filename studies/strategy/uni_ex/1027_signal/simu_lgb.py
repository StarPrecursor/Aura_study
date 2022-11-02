import logging
from concurrent.futures import ProcessPoolExecutor
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
model_config_path = Path("/home/yangzhe/Aura_study/model/fr_pred/1027_lgb/config.yaml")
with open(model_config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
model_dir = Path(cfg["model_dir"])


# Get inputs
print("Loading inputs...")
df = pd.read_feather(model_dir / "input.feather")
split_time = pd.to_datetime(cfg["train_test_split_date"])
df_tr, df_val, df_te = ar_model.train_utils.data_split_by_time(
    df, cfg["time"], split_time, val_ratio=0.2
)
t_tr = df_tr[cfg["time"]]
x_tr = df_tr[cfg["features"]]
y_tr = df_tr[cfg["target"]]
t_val = df_val[cfg["time"]]
x_val = df_val[cfg["features"]]
y_val = df_val[cfg["target"]]

t_te = df_te[cfg["time"]]
x_te = df_te[cfg["features"]]
y_te = df_te[cfg["target"]]
pr_te = df_te["price"]
fr_te = df_te["funding_rate"]


# Get predictions
model = ar_model.fr_model.FRModel_LGB(cfg)
model.load_model(model_dir, "model_epoch2000")
y_te_pred = model.predict(x_te)
df_te = df_te.assign(signal=y_te_pred)

# Plot y_te_pred hist
fig, ax = plt.subplots(figsize=(16, 9))
sns.histplot(y_te_pred, ax=ax)
ax.set_title("y_te_pred hist")
ax.set_xlabel("y_te_pred")
ax.set_ylabel("count")
fig.savefig("signal.png")

symbols = df_te["symbol"].unique()
trade_data = {}
for symbol in symbols:
    df_tmp = df_te[df_te["symbol"] == symbol]
    df_syb = pd.DataFrame()
    df_syb["time"] = df_tmp[cfg["time"]]
    df_syb["price"] = df_tmp["price"]
    df_syb["funding_rate"] = df_tmp["funding_rate"]
    df_syb["vol"] = df_tmp["vol"]
    df_syb["signal"] = df_tmp["signal"]
    trade_data[symbol] = df_syb


# ### Compare different capital sizes
def get_rec(cap):
    ts = ar_ana.simu.TradeSimulatorSignalSimple(trade_data)
    ts.set_buy_point(0.00002)
    ts.set_sell_point(0.00001)
    ts.trade(cap)
    plot_dir = Path(f"plot_lgb_cap_{int(cap / 1000)}k")
    plot_dir.mkdir(exist_ok=True)
    ts.plot_book(plot_dir)
    return ts

print("Start simulating...")
get_rec(1e6)
exit()


with ProcessPoolExecutor(max_workers=8) as executor:
    cap_list = [1e5, 1e6, 1e7]
    ts_lgb_list = executor.map(get_rec, cap_list)


fig, ax = plt.subplots(figsize=(8, 6))
for ts in ts_lgb_list:
    label = f"cap: {int(ts.cap / 1000)}k"
    ts.plot_pnl_cum_curve_rate(ax=ax, label=label)
ax.legend()
plot_dir = Path(f"plot_lgb_compare")
plot_dir.mkdir(exist_ok=True)
fig.savefig(plot_dir / "pnl_cum_curve_rate.png")


# ### Scan


scan_dim = {
    "sell_point": np.linspace(-0.0001, 0.0001, 11),
    "buy_point": np.linspace(-0.0001, 0.0001, 11),
}


def obj(sell_point, buy_point):
    ts = ar_ana.simu.TradeSimulatorSignalSimple(trade_data)
    ts.sell_point = sell_point
    ts.buy_point = buy_point
    ts.trade(1e6, show_progress=False)
    pnl = ts.trade_record["pnl"].sum()
    return pnl


df_scan = ar_ana.scan.grid_scan(scan_dim, obj)


# plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
df_tmp = df_scan.round(6).pivot("sell_point", "buy_point", "score")
# remove negative values
df_tmp[df_tmp < 0] = 0
sns.heatmap(df_tmp, ax=ax)
plot_dir = Path(f"plot_lgb_compare")
plot_dir.mkdir(exist_ok=True)
fig.savefig(plot_dir / "trade_point_scan.png")
