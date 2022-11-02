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

model_config_path = (
    Path("/home/yangzhe/Aura_study/model/fr_pred/1030_fut_n_trade")
    / f"config_fut_10.yaml"
)

with open(model_config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# Get inputs
print("Loading inputs...")
input_dir = Path(cfg["input_dir"])
df = pd.read_feather(input_dir / "input.feather")
split_time = pd.to_datetime(cfg["train_test_split_date"])
df_tr, _, df_te = ar_model.train_utils.data_split_by_time(
    df, cfg["time"], split_time, val_ratio=0
)
symbols = df_te["symbol"].unique()

score_rec = []
for ewm_hl in [8, 24, 48]:
    print("#" * 20)
    print(f"Using model fr ewm {ewm_hl}")
    # Study
    opt_buy = 0
    opt_sell = 0
    for data_cat, df_cur in [("train", df_tr), ("test", df_te)]:
        plot_dir = Path(f"ewm_{ewm_hl}/single_feature/{data_cat}")
        plot_dir.mkdir(parents=True, exist_ok=True)
        # Prepare trade data
        trade_data = {}
        for symbol in symbols:
            df_tmp = df_cur[df_cur["symbol"] == symbol]
            df_syb = pd.DataFrame()
            df_syb["time"] = df_tmp[cfg["time"]]
            df_syb["price"] = df_tmp["price"]
            df_syb["funding_rate"] = df_tmp["funding_rate"]
            df_syb["vol"] = df_tmp["vol"]
            if ewm_hl <= 24:
                df_syb["signal"] = df_tmp[f"funding_rate__ewm{ewm_hl}h"]
            else:
                dd = int(ewm_hl / 24)
                df_syb["signal"] = df_tmp[f"funding_rate__ewm{dd}d"]
            trade_data[symbol] = df_syb

        # Single simulation
        cap = 1e6
        ts = ar_ana.simu.TradeSimulatorSignalSimple(trade_data)
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
            print(f">> Optimal score: {opt_score}")
            test_train_diff = (opt_score - best_score) / best_score
            score_rec.append(
                [
                    opt_score,
                    test_train_diff,
                    ewm_hl,
                    None,
                    opt_buy,
                    opt_sell,
                    best_buy,
                    best_sell,
                    best_score,
                ]
            )

        # plot heatmap
        fig, ax = plt.subplots(figsize=(8, 7))
        df_tmp = df_scan.round(8).pivot("sell", "buy", "score")
        df_tmp[df_tmp < 0] = 0  # remove negative values
        sns.heatmap(df_tmp, ax=ax)
        fig.savefig(plot_dir / f"trade_scan_{data_cat}.png")

        # Single simulation
        cap = 1e6
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

score_df = pd.DataFrame(
    score_rec,
    columns=[
        "opt_score",
        "test_train_diff",
        "ewm_hl",
        "hl",
        "opt_buy",
        "opt_sell",
        "best_buy",
        "best_sell",
        "best_score",
    ],
)
score_df.to_csv("score_df_ewm.csv", index=False)

print("Top records:")
score_rec = sorted(score_rec, key=lambda x: x[0], reverse=True)
for i in range(min(10, len(score_rec))):
    print(score_rec[i])
