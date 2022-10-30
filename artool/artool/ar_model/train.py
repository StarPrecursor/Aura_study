import sys
from pathlib import Path

import pandas as pd
import yaml
from artool import ar_model
import logging

logger = logging.getLogger("artool")


def main():
    # get arguments from cmd line
    config_path = sys.argv[1]

    # Config
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model_dir = Path(cfg["model_dir"])
    model_name = cfg["model_name"]

    # Get inputs
    logger.info("Loading inputs...")
    input_dir = Path(cfg["input_dir"])
    df = pd.read_feather(input_dir / "input.feather")
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
    logger.info("Training model...")
    model = ar_model.fr_model.FRModel_LGB(cfg)
    model.set_inputs(x_tr, y_tr, x_val, y_val, x_te, y_te)
    model.train()
    model.save_model(model_dir, model_name)
