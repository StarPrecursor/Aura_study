import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("artool")


class TradeSimulatorBase:
    def __init__(self, trade_features=None):
        if trade_features is None:
            self.trade_features = [
                "price",
                "funding_rate",
                "vol",
                "signal",
            ]
        else:
            self.trade_features = trade_features


class TradeSimulatorSignal(TradeSimulatorBase):
    def __init__(self, trade_data: dict, fee: float = 3e-4):
        super().__init__()
        self.trade_data = trade_data
        self.symbols = list(trade_data.keys())
        self.hash_trade_data()
        self.get_ticks()
        self.fee = fee
        self.trade_record = None
        self.trade_done = False

    def get_ticks(self):
        ticks = None
        ticks_not_match = False
        for symbol in self.symbols:
            df = self.trade_data[symbol]
            if ticks is None:
                ticks = df["time"].values
            else:
                if np.all(ticks == df["time"].values):
                    continue
                else:
                    ticks = np.intersect1d(ticks, df["time"].values)
                    ticks_not_match = True
        if ticks_not_match:
            logger.warning("Time not match for all symbols")
        ticks = sorted(ticks)
        self.ticks = pd.to_datetime(ticks)

    def hash_trade_data(self):
        logger.info("Hashing trade data")
        trade_hash = {}
        for symbol in self.symbols:
            hash_dict = {}
            df = self.trade_data[symbol]
            for i, row in df.iterrows():
                tick = pd.to_datetime(row["time"])
                hash_dict[tick] = {
                    "price": row["price"],
                    "funding_rate": row["funding_rate"],
                    "vol": row["vol"],
                    "signal": row["signal"],
                }
            trade_hash[symbol] = hash_dict
        self.trade_hash = trade_hash

    def get_data(self, symbol, tick):
        return self.trade_hash[symbol][tick]

    def hash_trade_data_old(self):
        for symbol in self.symbols:
            df = self.trade_data[symbol]
            df = df.reset_index().set_index("time", drop=False)
            self.trade_data[symbol] = df

    def get_data_old(self, symbol, tick):
        df = self.trade_data[symbol]
        row = df.loc[tick]
        # row = df[df["time"] == tick].iloc[0]
        return {
            "price": row["price"],
            "funding_rate": row["funding_rate"],
            "vol": row["vol"],
            "signal": row["signal"],
        }

    def get_trade_record(self):
        if self.trade_done:
            return self.trade_record
        else:
            logger.error("Trade not done yet")
            return None

    def plot_book_base(self, save_dir):
        save_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_curve(ax=ax)
        fig.savefig(save_dir / "pnl_curve.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_hist(ax=ax)
        fig.savefig(save_dir / "pnl_hist.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_cum_curve(ax=ax)
        fig.savefig(save_dir / "pnl_cum_curve.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_ma_curve(ax=ax)
        fig.savefig(save_dir / "pnl_ma_curve.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_vol_curve(ax=ax)
        fig.savefig(save_dir / "vol_curve.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_n_symbol_curve(ax=ax)
        fig.savefig(save_dir / "n_symbol_curve.png")

    def plot_book(self, save_dir):
        self.plot_book_base(save_dir)

    def plot_pnl_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(data=self.trade_record, x="time", y="pnl", ax=ax, label=label)
        ax.set_title("PnL curve")
        return fig, ax

    def plot_pnl_hist(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(data=self.trade_record, x="pnl", ax=ax, label=label)
        ax.set_title("PnL hist")
        return fig, ax

    def plot_pnl_cum_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        df = self.trade_record
        df["pnl_cum"] = df["pnl"].cumsum()
        sns.lineplot(data=df, x="time", y="pnl_cum", ax=ax, label=label)
        ax.set_title("PnL cum curve")
        return fig, ax

    def plot_pnl_ma_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        df = self.trade_record
        df["pnl_ma"] = df["pnl"].expanding().mean()
        sns.lineplot(data=df, x="time", y="pnl_ma", ax=ax, label=label)
        ax.set_title("PnL ma curve")
        return fig, ax

    def plot_vol_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(data=self.trade_record, x="time", y="vol", ax=ax, label=label)
        ax.set_title("vol curve")
        return fig, ax

    def plot_vol_hist(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(data=self.trade_record, x="vol", ax=ax, label=label)
        ax.set_title("vol hist")
        return fig, ax

    def plot_n_symbol_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(data=self.trade_record, x="time", y="n_symbol", ax=ax, label=label)
        ax.set_title("n_symbol curve")
        return fig, ax


class TradeSimulatorSignalSimple(TradeSimulatorSignal):
    """Simulate trading with signal

    Input trade_data must contain:
        time, price, funding_rate, vol, signal

    Logic:
        - only consider positive rate profit
        - provide fixed amount of capital
        - sell:
            - sell symbols with signal < sell_point(default=0)
        - buy:
            - consider symbols with signal > buy_point(default=0)
            - buy top symbols with highest signal
        - each trade don't exceed 10% volume limits (buy/sell)

    """

    def __init__(self, trade_data, fee=10e-4):
        super().__init__(trade_data, fee=fee)
        self.cap = None
        self.sell_point = 0
        self.buy_point = 0

    def set_sell_point(self, sell_point):
        self.sell_point = sell_point

    def set_buy_point(self, buy_point):
        self.buy_point = buy_point

    def trade(self, cap, show_progress=True):
        self.cap = cap
        cap_free = cap
        trade_record = {
            "time": [],
            "cap_free": [],
            "cap_used": [],
            "n_symbol": [],
            "pnl": [],
            "vol": [],
        }
        share_holds = defaultdict(float)
        share_caps = defaultdict(float)
        logger.info("Simulating trading")
        cum_pnl = 0
        if show_progress:
            tq_ticks = tqdm(self.ticks)
        else:
            tq_ticks = self.ticks
        for tick in tq_ticks:
            pnl = 0
            cur_vol = 0
            # check if crypto trading hour (00:00, 08:00, 16:00)
            if pd.to_datetime(tick).hour in [0, 8, 16]:
                for symbol, hold in share_holds.items():
                    cur = self.get_data(symbol, tick)
                    pnl += hold * cur["price"] * cur["funding_rate"]
            # sell shares with negative signal
            for symbol, hold in share_holds.items():
                cur = self.get_data(symbol, tick)
                if cur["signal"] > self.sell_point or hold == 0:
                    continue
                # free capital
                reduced_cap = min(share_caps[symbol], cur["vol"] * 0.1)
                reduced_hold = reduced_cap / cur["price"]
                share_holds[symbol] -= reduced_hold
                share_caps[symbol] -= reduced_cap
                pnl -= reduced_cap * self.fee
                cap_free += reduced_cap
                cur_vol += reduced_cap
                logger.debug(f"sell {reduced_hold} {symbol} at {cur['price']}")
            # buy top positive signals
            sig_sybs = []
            for symbol in self.symbols:
                cur = self.get_data(symbol, tick)
                if cur["signal"] <= self.buy_point:
                    continue
                sig_sybs.append((cur["signal"], symbol))
            sig_sybs = sorted(sig_sybs, reverse=True)
            for _, symbol in sig_sybs:
                if cap_free <= 0.01:
                    break
                cur = self.get_data(symbol, tick)
                # buy
                cur_cap = min(cap_free, cur["vol"] * 0.1)
                new_hold = cur_cap / cur["price"]
                share_holds[symbol] += new_hold
                share_caps[symbol] += cur_cap
                pnl -= cur_cap * self.fee
                cap_free -= cur_cap
                cur_vol += cur_cap
                logger.debug(f"buy {new_hold} {symbol} at {cur['price']}")
            # record
            trade_record["time"].append(tick)
            trade_record["cap_free"].append(cap_free)
            trade_record["cap_used"].append(cap - cap_free)
            trade_record["pnl"].append(pnl)
            trade_record["vol"].append(cur_vol)
            n_symbol = 0
            for symbol, hold in share_holds.items():
                if hold > 0:
                    n_symbol += 1
            trade_record["n_symbol"].append(n_symbol)
            cum_pnl += pnl
            if show_progress:
                tq_ticks.set_description(f"cum_pnl: {cum_pnl:.2f} / {self.cap:.2f}")
        self.trade_record = pd.DataFrame(trade_record)
        self.trade_done = True

    def plot_book_simple(self, save_dir):
        save_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_cum_curve_rate(ax=ax)
        fig.savefig(save_dir / "pnl_cum_curve_rate.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_ma_curve_rate(ax=ax)
        fig.savefig(save_dir / "pnl_ma_curve_rate.png")

    def plot_book(self, save_dir):
        self.plot_book_base(save_dir)
        self.plot_book_simple(save_dir)

    def plot_pnl_cum_curve_rate(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        df = self.trade_record
        df["pnl_cum"] = df["pnl"].cumsum()
        df["pnl_cum_rate"] = df["pnl_cum"] / self.cap
        sns.lineplot(data=df, x="time", y="pnl_cum_rate", ax=ax, label=label)
        ax.set_title("PnL cum curve rate")
        return fig, ax

    def plot_pnl_ma_curve_rate(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        df = self.trade_record
        df["pnl_ma"] = df["pnl"].rolling(1000).mean()
        df["pnl_ma_rate"] = df["pnl_ma"] / self.cap
        sns.lineplot(data=df, x="time", y="pnl_ma_rate", ax=ax, label=label)
        ax.set_title("PnL ma curve rate")
        return fig, ax
