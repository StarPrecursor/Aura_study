import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import r2_score
import argparse
from artool import ar_ana, ar_model

matplotlib.use("Agg")  # use to improve performance
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("artool")
logging.basicConfig(level=logging.INFO)


def prepare_plot_dir(save_dir):
    plot_dir = Path(save_dir) / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    return plot_dir


def plot_learning_curve(model, save_dir):
    logger.info("Plotting learning curve...")
    plot_dir = prepare_plot_dir(save_dir)
    eval_result = model.evals_result
    fig, ax = plt.subplots()
    for metric, data in eval_result.items():
        for key, values in data.items():
            ax.plot(values, label=f"{key}-{metric}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("score")
    ax.legend()
    fig.savefig(plot_dir / "learning_curve.png")
    plt.close(fig)


def plot_pred_vs_true(y_true, y_pred, save_dir, label=""):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Check Nan
    n_nan = np.isnan(y_true).sum()
    if n_nan > 0:
        logger.warning(f"y_true has {n_nan} NaNs.")
    n_nan = np.isnan(y_pred).sum()
    if n_nan > 0:
        logger.warning(f"y_pred has {n_nan} NaNs.")
    # Ignore Nan
    logger.warning("Ignore NaNs.")
    nan_idx = np.isnan(y_true) | np.isnan(y_pred)
    y_true = y_true[~nan_idx]
    y_pred = y_pred[~nan_idx]
    # Plot
    logger.info(f"Plotting pred vs true ... ({label})")
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    plot_dir = prepare_plot_dir(save_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.1)
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    ax.set_title(f"{label} - r2={r2:.3f}, corr={corr:.3f}")
    fig.savefig(plot_dir / f"pred_vs_true.{label}.png")
    plt.close(fig)


def plot_symbol_pred_curve(df, t, y_pred, save_dir, label, chunk_size=5, y_true=None):
    logger.info(f"Plotting symbol pred curve ... ({label})")
    df = df.copy()
    symbols = df["symbol"].unique()
    plot_dir = save_dir / "plots" / "symbols"
    plot_dir.mkdir(exist_ok=True, parents=True)
    for symbols_sub in np.array_split(symbols, len(symbols) // chunk_size + 1):
        # plot weekly
        df["wk_n"] = pd.to_datetime(df[t]).dt.isocalendar().week.astype(int)
        wk_n_list = df["wk_n"].unique()
        for wk_n in wk_n_list:
            df_mm = df[df["wk_n"] == wk_n]
            fig, ax = plt.subplots(figsize=(18, 6))
            for symbol in symbols_sub:
                df_sub = df_mm[df_mm["symbol"] == symbol]
                ax.plot(df_sub[t], df_sub[y_pred], label=symbol)
            ax.legend()
            plot_dir_sub = plot_dir / f"week_{wk_n}"
            plot_dir_sub.mkdir(exist_ok=True, parents=True)
            fig.savefig(
                plot_dir_sub
                / f"{symbols_sub[0]}_{symbols_sub[-1]}_pred_curve.{label}.png"
            )
            plt.close(fig)
            # plot true target if specified
            if y_true is not None:
                fig, ax = plt.subplots(figsize=(18, 6))
                for symbol in symbols_sub:
                    df_sub = df_mm[df_mm["symbol"] == symbol]
                    ax.plot(df_sub[t], df_sub[y_true], label=symbol)
                ax.legend()
                plot_dir_sub = plot_dir / f"week_{wk_n}"
                plot_dir_sub.mkdir(exist_ok=True, parents=True)
                fig.savefig(
                    plot_dir_sub
                    / f"{symbols_sub[0]}_{symbols_sub[-1]}_true_curve.{label}.png"
                )
                plt.close(fig)


def shap_summary_plot(ar_model, X, save_dir, label, sampling=None):
    shap_exp = shap.TreeExplainer(ar_model.model)
    if sampling is not None:
        X = X.sample(sampling)
    shap_values = shap_exp.shap_values(X)
    plt.clf()
    shap.summary_plot(shap_values, X, show=False)
    shap_dir = Path(save_dir) / "shap"
    shap_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(shap_dir / f"shap_summary_plot.{label}.png")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="*", action="store", help="config path(s) for model evaluation")
    parser.add_argument(
        "-n", "--n_sample", type=int, default=None, help="number of samples"
    )
    args = parser.parse_args()

    if not args.config_path:
        logger.error("No config path provided")
        parser.print_help()
        exit()
    else:
        for p in args.config_path:
            if not Path(p).exists():
                logger.error(f"Config path {p} does not exist, skipping")
                continue
            logger.info("#"*80)
            logger.info(f"Executing config: {p}")
            # Config
            with open(p) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            model_dir = Path(cfg["model_dir"])
            model_name = cfg["model_name"]

            # Load model
            logger.info("Loading model...")
            model = ar_model.fr_model.FRModel_LGB(cfg)
            model.load_model(model_dir, model_name)

            # Get inputs
            logger.info("Loading inputs...")
            input_dir = Path(cfg["input_dir"])
            df = pd.read_feather(input_dir / "input.feather")
            df["y_pred"] = model.predict(df[cfg["features"]])
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

            # Evaluate
            logger.info("Evaluating...")
            if "eva_name" in cfg:
                save_dir = model_dir / cfg["eva_name"]
            else:
                save_dir = model_dir
            save_dir.mkdir(exist_ok=True, parents=True)
            plot_learning_curve(model, save_dir)
            y_tr_pred = df_tr["y_pred"]
            y_val_pred = df_val["y_pred"]
            y_te_pred = df_te["y_pred"]
            plot_pred_vs_true(y_tr, y_tr_pred, save_dir, label="train")
            plot_pred_vs_true(y_val, y_val_pred, save_dir, label="val")
            plot_pred_vs_true(y_te, y_te_pred, save_dir, label="test")

            # Plot symbol pred curve
            plot_symbol_pred_curve(
                df_te, cfg["time"], "y_pred", save_dir, label="test", y_true=cfg["target"]
            )

            # SHAP
            logger.info("SHAP importance study...")
            shap_summary_plot(model, x_tr, save_dir, "train", sampling=args.n_sample)
            shap_summary_plot(model, x_val, save_dir, "val", sampling=args.n_sample)
            shap_summary_plot(model, x_te, save_dir, "test", sampling=args.n_sample)
