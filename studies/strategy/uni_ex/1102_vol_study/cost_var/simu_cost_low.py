import logging
from pathlib import Path

import pandas as pd
import yaml
from artool import ar_ana, ar_io, ar_model

# set logging level
logging.basicConfig(level=logging.INFO)
logging.getLogger("artool").setLevel(logging.INFO)

# Trade parameters
fut = 10
opt_buy = 0.00007
opt_sell = -0.00009
fee = 3e-4

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

# Single simulation
for data_cat, df_cur in [("train", df_tr), ("test", df_te)]:
    plot_dir = Path(data_cat)
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

    # simu
    cap = 1e6
    ts = ar_ana.simu.TradeSimulatorSignalSimple(trade_data)
    ts.vol_lim = 0.05
    ts.hold_lim = 0.2
    ts.fee = fee
    ts.set_buy_point(opt_buy)
    ts.set_sell_point(opt_sell)
    ts.trade(cap)
    ts.plot_book(plot_dir)

    profit_rate_per_year = ts.get_total_pnl() / cap
    if data_cat == "train":
        print(f"Train profit rate per year: {profit_rate_per_year * 12 / 5}")
    elif data_cat == "test":
        print(f"Test profit rate per year: {profit_rate_per_year * 12 / 3}")
