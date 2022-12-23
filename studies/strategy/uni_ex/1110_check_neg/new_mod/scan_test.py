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

# parameters setting
def_para = {
    "vol_lim": 0.05,
    "hold_lim": 0.2,
    "fee": 10e-4,
    "cap": 1e6,
}
use_strategy = ar_ana.strategies.ChampagneTowerNegative

# Get inputs
print("Loading inputs...")
input_dir = Path("/home/yangzhe/model/fr_pred/1110_neg_check")
mod_cfg_dir = Path("/home/yangzhe/Aura_study/model/fr_pred/1110_mod_Jun_Oct")
df = pd.read_feather(input_dir / "input.feather")
symbols = df["symbol"].unique()

score_rec = []
for fut in [1, 3, 5, 10, 15, 21, 30, 45]:
#for fut in [1, 3, 5, 10]:
# for fut in [15, 21, 30, 45]:
    print("#" * 20)
    print(f"Using model fut {fut}")
    # Config
    model_config_path = mod_cfg_dir / f"config_fut_{fut}.yaml"
    with open(model_config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Predict
    print("Predicting...")
    model_dir = Path(cfg["model_dir"])
    model = ar_model.fr_model.FRModel_LGB(cfg)
    model.load_model(model_dir, cfg["model_name"])
    pred = model.predict(df[cfg["features"]])
    df["signal"] = pred

    # Study
    df_cur = df
    plot_dir = Path(f"fut_{fut}/test")
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
            "amount": "amount",
            "interest": "interest",
        },
    )
    ts = ar_ana.simu.TradeSimulatorSignalGeneralNegtive(trade_data)
    ts.set_strategy(use_strategy)
    ts.update_para(def_para)

    def get_score(buy, sell, ts=ts, plot=False, show_progress=False):
        ts_cur = copy.copy(ts)
        if not show_progress:
            ts_cur.strategy.show_progress = False
        ts_cur.set_buy_point(buy)
        ts_cur.set_sell_point(sell)
        ts_cur.trade()
        if plot:
            ts_cur.plot_book(plot_dir)
        # return ts_cur.get_total_pnl()
        profit_rate_per_year = ts_cur.get_total_pnl() / def_para["cap"]
        return profit_rate_per_year * 365 / 20

    # scan
    scan_dim = {
        "buy": np.linspace(-5e-4, 0, 21),
        "sell": np.linspace(-4e-4, 1e-4, 21),
    }
    df_scan = ar_ana.scan.grid_scan(scan_dim, get_score)

    # find sell/buy point of highest score
    top_idx = df_scan["score"].idxmax()
    best_buy = df_scan.loc[top_idx, "buy"]
    best_sell = df_scan.loc[top_idx, "sell"]
    best_score = df_scan.loc[top_idx, "score"]
    print(f"* Optimal buy point: {best_buy}")
    print(f"* Optimal sell point: {best_sell}")
    opt_score = get_score(best_buy, best_sell, plot=True, show_progress=True)
    print(f"#### > Optimal score: {opt_score} < ####")
    score_rec.append(
        [
            opt_score,
            fut,
            best_buy,
            best_sell,
        ]
    )

    # plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    df_tmp = df_scan.round(8).pivot("sell", "buy", "score")
    df_tmp[df_tmp < 0] = 0  # remove negative values
    sns.heatmap(df_tmp, ax=ax)
    fig.savefig(plot_dir / f"trade_scan_test.png")


score_df = pd.DataFrame(
    score_rec,
    columns=[
        "opt_score",
        "fut",
        "best_buy",
        "best_sell",
    ],
)
score_df.to_csv("score_df.csv", index=False)
print("Top records:")
score_rec = sorted(score_rec, key=lambda x: x[0], reverse=True)
for i in range(min(10, len(score_rec))):
    print(score_rec[i])
