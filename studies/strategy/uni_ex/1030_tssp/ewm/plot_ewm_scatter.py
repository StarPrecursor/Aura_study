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
model_config_path = Path("/home/yangzhe/Aura_study/model/fr_pred/1030_fut_n_trade/config_fut_5.yaml")
with open(model_config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Get inputs
print("Loading inputs...")
input_dir = Path(cfg["input_dir"])
df = pd.read_feather(input_dir / "input.feather")
split_time = pd.to_datetime(cfg["train_test_split_date"])

y_pred = df["funding_rate__ewm2d"]
y_true = df["funding_rate_trade__future_5"]
idx_nan = y_true.isna() | y_pred.isna()
y_true = y_true[~idx_nan]
y_pred = y_pred[~idx_nan]

# plto scatter
fig, ax = plt.subplots()
ax.scatter(y_true, y_pred, s=1)
ax.set_xlabel("y_true")
ax.set_ylabel("ewm_2d")
r2 = r2_score(y_true, y_pred)
corr = np.corrcoef(y_true, y_pred)[0, 1]
ax.set_title(f"r2: {r2:.3f}, corr: {corr:.3f}")
fig.savefig("y_true_vs_ewm2d.png")
