

import logging
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import r2_score

matplotlib.use("Agg")  # use to improve performance
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("artool")


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
    plot_dir = prepare_plot_dir(save_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.1)
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    ax.set_title(f"{label}-r2: {r2:.3f}")
    fig.savefig(plot_dir / f"pred_vs_true.{label}.png")
    plt.close(fig)
