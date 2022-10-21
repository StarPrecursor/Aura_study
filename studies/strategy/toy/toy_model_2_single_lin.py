import datetime
import multiprocessing as mp
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from artool import toy
from artool.toy.toy_simu import get_pnl_simple

# remove limits on number of rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# ## Settings
rdm_seed = 42
max_workers = 36
data_dir = Path("/home/yangzhe/data/toy_data_2")
date_start = datetime.datetime(2022, 3, 1)
date_end = datetime.datetime(2022, 9, 1)
symbols = toy.toy_data.get_symbol_list(date_start, date_end)
make_plot = False

target_symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT"]
print(f"target_symbols: {target_symbols}")


# Use columns from `features`
symbol = target_symbols[0]
df = pd.read_feather(data_dir / f"{symbol}.feather")
features = df.columns.tolist()
features = [x for x in features if not x.startswith("funding_rate_future")]
x_features = features[2:]

# Input Processing
targets = [x for x in df.columns if x.startswith("funding_rate_future")]
print(targets)

train_end_date = datetime.datetime(2022, 7, 1)
train_idx_dict = {}
for symbol in target_symbols:
    df = pd.read_feather(data_dir / f"{symbol}.feather")
    #train_idx = pd.to_datetime(df["funding_timestamp"], unit="us") < train_end_date

    # get train index before train_end_date
    train_idx = df["funding_timestamp"] < train_end_date.timestamp() * 1e6
    train_idx_dict[symbol] = train_idx



print(f"#### training single variant linear regression")
fig_dir = Path("./toy_model_2/figures/lin_sin")
fig_dir.mkdir(exist_ok=True, parents=True)
corr_dir = fig_dir / "corr"
corr_dir.mkdir(exist_ok=True, parents=True)
scan_dir = fig_dir / "pnl_scan"
scan_dir.mkdir(exist_ok=True, parents=True)

plot_tasks = []
for symbol in target_symbols:
    print(f"Processing {symbol}")
    train_idx = train_idx_dict[symbol]
    test_idx = ~train_idx
    # set train_idx first 50 to False
    train_idx.iloc[:50] = False

    df = pd.read_feather(data_dir / f"{symbol}.feather")
    funding_rate = df["funding_rate"]
    fr_train = funding_rate[train_idx]
    fr_test = funding_rate[test_idx]

    for f in x_features:

        x = df[[f]].values
        x_train_all = x[train_idx]
        x_test = x[test_idx]
        y = df["funding_rate_future_5"].fillna(0).values
        y_train_all = y[train_idx]
        y_test = y[test_idx]

        # train validation split
        x_train, x_val, y_train, y_val, fr_train, fr_val = train_test_split(
            x_train_all, y_train_all, fr_train, test_size=0.2, random_state=42
        )
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train, free_raw_data=False)
        lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train, free_raw_data=False)

        model = LinearRegression()
        model.fit(x_train, y_train)

        print("predicting")
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        # plot y_pred vs y
        if make_plot:
            print("plotting")
            fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
            ax = np.array(ax)
            ax[0].scatter(y_train, y_pred_train, label="train")
            r2_train = r2_score(y_train, y_pred_train)
            # ax[0].set_title(f"train, r2: {r2_train:.3f}")
            corr_train = np.corrcoef(y_train, y_pred_train)[0, 1]
            ax[0].set_title(f"train, r2: {r2_train:.3f}, corr: {corr_train:.3f}")

            ax[0].set_xlabel("funding_rate_future_5")
            ax[0].set_ylabel("y_pred")

            ax[1].scatter(y_test, y_pred_test, label="test")
            r2_test = r2_score(y_test, y_pred_test)
            # ax[1].set_title(f"test")
            corr_test = np.corrcoef(y_test, y_pred_test)[0, 1]
            ax[1].set_title(f"test, r2: {r2_test:.3f}, corr: {corr_test:.3f}")

            ax[1].set_xlabel("funding_rate_future_5")
            ax[1].set_ylabel("y_pred")

            fig_dir = corr_dir / f"{f}"
            fig_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(corr_dir / f"{symbol}_lgb.png")  # type: ignore
            plt.close(fig)

        # plt.show()
        # break

        # find optimal pnl
        y_min = min(y_pred_test.min(), y_pred_train.min())
        y_max = max(y_pred_test.max(), y_pred_train.max())
        buy_thre_list = np.linspace(y_min, y_max, 21)
        sell_thre_list = np.linspace(y_min, y_max, 21)
        pnl_list_train = [[0] * 21 for _ in range(21)]
        pnl_list_test = [[0] * 21 for _ in range(21)]
        for i, buy_thre in enumerate(buy_thre_list):
            for j, sell_thre in enumerate(sell_thre_list):
                pnl_list_train[i][j] = get_pnl_simple(
                    y_pred_train, fr_train, buy_thre, sell_thre
                )
                pnl_list_test[i][j] = (
                    get_pnl_simple(y_pred_test, fr_test, buy_thre, sell_thre) * 3
                )

        if make_plot:

            # plot in a new process
            def plot_pnl(pnl_list_train, pnl_list_test, save_path):
                fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
                ax = np.array(ax)
                sns.heatmap(pnl_list_train, ax=ax[0])
                sns.heatmap(pnl_list_test, ax=ax[1])
                ax[0].set_title("train")
                ax[0].set_xlabel("buy_thre")
                ax[0].set_ylabel("sell_thre")
                ax[1].set_title("test x3")
                ax[1].set_xlabel("buy_thre")
                ax[1].set_ylabel("sell_thre")
                fig.suptitle(f"{symbol}")
                fig.savefig(save_path)

            p = mp.Process(
                target=plot_pnl,
                args=(pnl_list_train, pnl_list_test, corr_dir / f"{symbol}_lgb.png"),
            )
            p.start()
            plot_tasks.append(p)

        # plt.show()
        # break

        # find location of max pnl in train
        max_i, max_j = np.unravel_index(np.argmax(pnl_list_train), (21, 21))
        max_pnl_train = pnl_list_train[max_i][max_j]
        pnl_test_opt = pnl_list_test[max_i][max_j]
        print(f"max_pnl in test: {pnl_test_opt}")

        break
    break

# wait for plot_tasks to finish
print("waiting for plot_tasks to finish")
for p in plot_tasks:
    p.join()
