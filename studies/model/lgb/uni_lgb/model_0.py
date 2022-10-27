import datetime
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import artool
from artool import ar_ana, toy
from artool.toy.toy_simu import get_pnl_2side, get_pnl_simple

# remove limits on number of rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import matplotlib

matplotlib.use("Agg")  # use to improve performance
import matplotlib.pyplot as plt
import seaborn as sns

# Config
config = yaml.safe_load(open("model_0.yaml"))

# Data
data_dir = Path("/home/yangzhe/data/toy_data_2")
date_start = datetime.datetime(2022, 1, 1)
date_end = datetime.datetime(2022, 9, 1)
symbols = toy.toy_data.get_symbol_list(date_start, date_end, logic="and")
#symbols = config["symbols"]
logging.info(f"Number of symbols: {len(symbols)}")
symbol_id_dict = {symbol: i for i, symbol in enumerate(symbols)}

logging.info(f"Loading data...")
df = pd.DataFrame()
for symbol in symbols:
    df_ = pd.read_feather(data_dir / f"{symbol}.feather")

    # add lagging features
    df_["funding_rate__lag1"] = df_["funding_rate"].shift(1)
    df_["funding_rate__lag3"] = df_["funding_rate"].shift(3)
    df_["funding_rate__lag5"] = df_["funding_rate"].shift(5)

    # add differences
    df_["funding_rate__diff1"] = df_["funding_rate"].shift(1) - df_["funding_rate"]
    df_["funding_rate__diff3"] = df_["funding_rate"].shift(3) - df_["funding_rate"]
    df_["funding_rate__diff5"] = df_["funding_rate"].shift(5) - df_["funding_rate"]

    df_["price_diff"] = df_["mark_price"] - df_["index_price"]

    # remove head/tail 5 rows
    df_ = df_.iloc[5:-5]
    df = pd.concat([df, df_], axis=0)
df["symbol_id"] = df["symbol"].map(symbol_id_dict)
df = df.reset_index(drop=True)

# Engineer features

## datetime
dt_values = pd.to_datetime(df["funding_timestamp"], unit="us")
df["dt__day_of_week"] = dt_values.dt.dayofweek
df["dt__day_of_month"] = dt_values.dt.day
df["dt__hour"] = dt_values.dt.hour
df["dt__is_quarter_start"] = dt_values.dt.is_quarter_start
df["dt__is_quarter_end"] = dt_values.dt.is_quarter_end
df["dt__is_month_start"] = dt_values.dt.is_month_start
df["dt__is_month_end"] = dt_values.dt.is_month_end
df["dt__is_weekend"] = dt_values.dt.dayofweek >= 5
cld = USFederalHolidayCalendar()
holidays = cld.holidays(start=dt_values.min(), end=dt_values.max())
df["dt__is_holiday"] = dt_values.dt.date.astype("datetime64").isin(holidays)
## extra
df_fear_greed = pd.read_feather("/home/yangzhe/data/external_data/fear_and_greed_index.feather")
df_fear_greed["date"] = pd.to_datetime(df_fear_greed["timestamp"]).dt.date
df_fear_greed["fear_and_greed_value"] = df_fear_greed["fear_and_greed_value"].shift(1)
df_fear_greed["fear_and_greed_class"] = df_fear_greed["fear_and_greed_class"].shift(1)
df["date"] = dt_values.dt.date
df = df.merge(df_fear_greed[["date", "fear_and_greed_value", "fear_and_greed_class"]], on="date", how="left")

## train / test separation
train_end_date = datetime.datetime(2022, 6, 1)
train_idx = df["funding_timestamp"] < train_end_date.timestamp() * 1e6
test_idx = ~train_idx
df_train = df[train_idx]
df_test = df[test_idx]

features = config["features"]
target_feature = config["target"]
x_tr, x_val, y_tr, y_val, fr_tr, fr_val = train_test_split(
    df_train[features],
    df_train[target_feature],
    df_train["funding_rate"],
    test_size=0.2,
    random_state=42,
)
x_te = df_test[features]
y_te = df_test[target_feature]

lgb_train = lgb.Dataset(x_tr, y_tr, categorical_feature=config["categorical_feature"])
lgb_eval = lgb.Dataset(
    x_val, y_val, categorical_feature=config["categorical_feature"], reference=lgb_train
)

res_dir = Path("./model_0")
res_dir.mkdir(exist_ok=True)

# Model
logging.info(f"Training model...")
model = lgb.train(
    config["params"],
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    **config["train_kargs"],
)
# save model
n_boost = config["train_kargs"]["num_boost_round"]
model.save_model(str(res_dir / f"model_{n_boost}.txt"))


# Evaluation
y_pred_tr = model.predict(x_tr)
y_pred_te = model.predict(x_te)
df_train = df_train.assign(y_pred=model.predict(df_train[features]))
df_test = df_test.assign(y_pred=y_pred_te)
#df_train["y_pred"] = model.predict(df_train[features])
#df_test["y_pred"] = y_pred_te
logging.info(f"Evaluating...")
r2_train = r2_score(y_tr, y_pred_tr)
corr_train = np.corrcoef(y_tr, y_pred_tr)[0, 1]
r2_test = r2_score(y_te, y_pred_te)
corr_test = np.corrcoef(y_te, y_pred_te)[0, 1]



fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
ax = np.array(ax)
ax[0].scatter(y_tr, y_pred_tr, alpha=0.1, label="train")
ax[0].set_title(f"train, r2: {r2_train:.3f}, corr: {corr_train:.3f}")
ax[0].set_xlabel("funding_rate_future_5")
ax[0].set_ylabel("y_pred")
ax[1].scatter(y_te, y_pred_te, alpha=0.1, label="test")
ax[1].set_title(f"test, r2: {r2_test:.3f}, corr: {corr_test:.3f}")
ax[1].set_xlabel("funding_rate_future_5")
ax[1].set_ylabel("y_pred")
fig.savefig(res_dir / f"corr_scatter.png")
plt.close(fig)

# Check r2_score in each symbol
symbol_score = {}
for symbol in symbols:
    idx = df_train["symbol"] == symbol
    r2_tr = r2_score(y_tr.loc[idx], model.predict(x_tr.loc[idx]))
    idx = df_test["symbol"] == symbol
    r2_te = r2_score(y_te.loc[idx], model.predict(x_te.loc[idx]))
    symbol_score[symbol] = (r2_tr, r2_te)

# Soret by r2_score and print
symbol_score = sorted(symbol_score.items(), key=lambda x: x[1][0], reverse=True)
for symbol, (r2_tr, r2_te) in symbol_score:
    print(f"{symbol}: tr - {r2_tr:.3f}, te - {r2_te:.3f}")
# Save top symbols to yaml
top_symbols = [symbol for symbol, (r2_tr, _) in symbol_score if r2_tr >= 0.4]
with open(res_dir / "top_symbols.yaml", "w") as f:
    yaml.dump(top_symbols, f)

corr_dir = res_dir / "corr"
corr_dir.mkdir(exist_ok=True)
for i, (symbol, _) in enumerate(symbol_score[:20]):
    #symbol = symbol_score[0][0]
    df_tr_tmp = df_train[df_train["symbol"] == symbol]
    y_tr_syb = df_tr_tmp[target_feature]
    y_pred_tr_syb = df_tr_tmp["y_pred"]
    r2_train = r2_score(y_tr_syb, y_pred_tr_syb)

    df_te_tmp = df_test[df_test["symbol"] == symbol]
    y_te_syb = df_te_tmp[target_feature]
    y_pred_te_syb = df_te_tmp["y_pred"]
    r2_test = r2_score(y_te_syb, y_pred_te_syb)

    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    ax = np.array(ax)
    ax[0].scatter(y_tr_syb, y_pred_tr_syb, alpha=0.1, label="train")
    ax[0].set_title(f"train, r2: {r2_train:.3f}, corr: {corr_train:.3f}")
    ax[0].set_xlabel("funding_rate_future_5")
    ax[0].set_ylabel("y_pred")
    ax[1].scatter(y_te_syb, y_pred_te_syb, alpha=0.1, label="test")
    ax[1].set_title(f"test, r2: {r2_test:.3f}, corr: {corr_test:.3f}")
    ax[1].set_xlabel("funding_rate_future_5")
    ax[1].set_ylabel("y_pred")
    fig.suptitle(symbol)
    fig.savefig(corr_dir / f"corr_scatter_top_{i}.png")
    plt.close(fig)
