from pathlib import Path

import pandas as pd
import yaml

from artool import ar_model

#model_name = "model_debug"
model_name = "model_epoch2000"

# Config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
model_dir = Path(cfg["model_dir"])

# Get inputs
print("Loading inputs...")
df = pd.read_feather(model_dir / "input.feather")
split_time = pd.to_datetime(cfg["train_test_split_date"])
df_tr, df_val, df_te = ar_model.train_utils.data_split_by_time(
    df, cfg["time"], split_time, val_ratio=0.2
)
x_tr = df_tr[cfg["features"]]
y_tr = df_tr[cfg["target"]]
x_val = df_val[cfg["features"]]
y_val = df_val[cfg["target"]]
x_te = df_te[cfg["features"]]
y_te = df_te[cfg["target"]]

# Train
print("Training model...")
model = ar_model.fr_model.FRModel_LGB(cfg)
model.set_inputs(x_tr, y_tr, x_val, y_val, x_te, y_te)
model.train()
model.save_model(model_dir, model_name)

print(model.evals_result)
