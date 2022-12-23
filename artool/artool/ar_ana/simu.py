import logging
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

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

    def update_para(self, para_dict: dict):
        for k, v in para_dict.items():
            setattr(self, k, v)


class TradeSimulatorSignal(TradeSimulatorBase):
    def __init__(
        self,
        trade_data: dict,
        fee: float = 10e-4,
        preprocess=True,
        mp_hash=True,
        trade_features=None,
    ):
        super().__init__(trade_features=trade_features)
        self.trade_data = trade_data
        self.trade_hash = None
        self.symbols = list(trade_data.keys())
        if preprocess:
            self.hash_trade_data(mp=mp_hash)
            self.get_ticks()
        self.fee = fee
        self.trade_record = None
        self.trade_done = False
        self.signal_true = False
        if "signal_true" in self.trade_data[self.symbols[0]].columns:
            self.signal_true = True
        self.trade_only_8h = False

    def __copy__(self):
        new_self = TradeSimulatorSignal(self.trade_data, preprocess=False)
        new_self.__dict__.update(self.__dict__)
        new_self.trade_record = None
        new_self.trade_done = False
        return new_self

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

    def hash_trade_data(self, trade_hash=None, show_progress=True, mp=True):
        logger.info("Hashing trade data")
        if trade_hash is not None:
            self.trade_hash = trade_hash
            return
        if mp:
            self.hash_trade_data_mp()
            return
        trade_hash = {}
        if show_progress:
            tq_iter = tqdm(self.symbols)
        else:
            tq_iter = self.symbols
        for symbol in tq_iter:
            _, trade_hash[symbol] = self.hash_trade_data_worker(symbol)
        self.trade_hash = trade_hash

    def hash_trade_data_worker(self, symbol):
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
        return symbol, hash_dict

    def hash_trade_data_mp(self):
        """ "Hash trade data with multi-process approach"""
        trade_hash = {}
        with ProcessPoolExecutor() as executor:
            for symbol, hash_dict in executor.map(
                self.hash_trade_data_worker, self.symbols
            ):
                trade_hash[symbol] = hash_dict
        self.trade_hash = trade_hash

    def get_data(self, symbol, tick):
        return self.trade_hash[symbol][tick]

    def hash_trade_data_old(self, show_progress=True):
        if show_progress:
            tq_iter = tqdm(self.symbols)
        else:
            tq_iter = self.symbols
        for symbol in tq_iter:
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

    def get_total_pnl(self):
        if self.trade_done:
            return self.trade_record["pnl"].sum()
        else:
            logger.error("Trade not done yet")
            return None

    def plot_book_base(self, save_dir):
        save_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(16, 9))
        self.plot_signal_hist(fig, ax)
        fig.savefig(save_dir / "signal_hist.png")
        fig, ax = plt.subplots(figsize=(16, 9))
        self.plot_signal_scatter(fig, ax)
        fig.savefig(save_dir / "signal_scatter.png")
        if self.signal_true:
            fig, ax = plt.subplots(figsize=(16, 9))
            self.plot_signal_scatter(y="signal_true", fig=fig, ax=ax)
            fig.savefig(save_dir / "signal_true_scatter.png")
        # PnL
        pnl_dir = save_dir / "pnl"
        pnl_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_pnl_curve(ax=ax)
        fig.savefig(pnl_dir / "pnl_curve.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_pnl_hist(ax=ax)
        fig.savefig(pnl_dir / "pnl_hist.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_pnl_cum_curve(ax=ax)
        fig.savefig(pnl_dir / "pnl_cum_curve.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_pnl_ma_curve(ax=ax)
        fig.savefig(pnl_dir / "pnl_ma_curve.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_pnl_fee_relation(ax=ax)
        fig.savefig(pnl_dir / "pnl_fee_relation.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_market_trend(ax=ax)
        fig.savefig(pnl_dir / "market_trend.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_market_trend_ratio(ax=ax)
        fig.savefig(pnl_dir / "market_trend_ratio.png")
        # Volume
        vol_dir = save_dir / "vol"
        vol_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_vol_curve(ax=ax)
        fig.savefig(vol_dir / "vol_curve.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_vol_hist(ax=ax)
        fig.savefig(vol_dir / "vol_hist.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_vol_hour_hist(ax=ax)
        fig.savefig(vol_dir / "vol_hour_hist.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_cap_curve(ax=ax)
        fig.savefig(vol_dir / "cap_curve.png")
        # Symbol
        fig, ax = plt.subplots(figsize=(12, 6))
        symbol_dir = save_dir / "symbol"
        symbol_dir.mkdir(exist_ok=True)
        self.plot_n_symbol_curve(ax=ax)
        fig.savefig(symbol_dir / "n_symbol_curve.png")

    def plot_book(self, save_dir):
        if not self.trade_done:
            logger.error("Trade not done yet, will not make plot book")
            return
        self.plot_book_base(save_dir)
        # close all figures
        plt.close("all")

    def plot_signal_hist(self, fig=None, ax=None, label=None):
        y_list = []
        for _, df in self.trade_data.items():
            y_list.append(df["signal"].values)
        y = np.concatenate(y_list)
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(y, bins=50, label=label)
        ax.set_xlabel("Signal")
        ax.set_ylabel("Count")
        return fig, ax

    def plot_signal_scatter(self, fig=None, ax=None, label=None, y="signal"):
        t_list = []
        y_list = []
        for _, df in self.trade_data.items():
            t_list.append(df["time"].values)
            y_list.append(df[y].values)
        t = np.concatenate(t_list)
        y = np.concatenate(y_list)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(t, y, s=1, alpha=0.2, label="pred")
        ax.axhline(self.buy_point, color="red", linestyle="--", alpha=0.5, label="buy")
        ax.axhline(
            self.sell_point, color="green", linestyle="--", alpha=0.5, label="sell"
        )
        patience = abs(self.buy_point - self.sell_point)
        y_max = max(self.buy_point, self.sell_point) + patience / 2
        y_min = min(self.buy_point, self.sell_point) - patience
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal")
        ax.legend()
        return fig, ax

    def plot_market_trend(self, fig=None, ax=None, label=None):
        t_list = []
        fr_list = []
        vol_list = []
        for _, df in self.trade_data.items():
            t_list.append(df["time"].values)
            fr_list.append(df["funding_rate"].values)
            vol_list.append(df["vol"].values)
        t = np.concatenate(t_list)
        fr = np.concatenate(fr_list)
        vol = np.concatenate(vol_list)
        fr_vol = fr * vol
        t_unique, t_idx = np.unique(t, return_inverse=True)
        fr_vol_sum = np.zeros_like(t_unique, dtype=np.float64)
        fr_vol_pos = np.zeros_like(t_unique, dtype=np.float64)
        fr_vol_neg = np.zeros_like(t_unique, dtype=np.float64)
        for i in range(len(t_unique)):
            fr_vol_sum[i] = np.sum(fr_vol[t_idx == i])
            fr_vol_pos[i] = np.sum(fr_vol[t_idx == i][fr_vol[t_idx == i] > 0])
            fr_vol_neg[i] = np.sum(fr_vol[t_idx == i][fr_vol[t_idx == i] < 0])
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        scale = np.mean(np.abs(fr_vol_sum))
        fr_vol_sum /= scale
        fr_vol_pos /= scale
        fr_vol_neg /= scale

        ax.plot(t_unique, fr_vol_sum, label="all", alpha=0.5, color="grey")
        ax.plot(t_unique, fr_vol_pos, label="pos", alpha=0.5, color="red")
        ax.plot(t_unique, fr_vol_neg, label="neg", alpha=0.5, color="green")
        # plot moving average
        # ma_24 = np.convolve(fr_vol_sum, np.ones(24) / 24, mode="same")
        # ma_72 = np.convolve(fr_vol_sum, np.ones(72) / 72, mode="same")
        # ax2 = ax.twinx()
        # ax2.plot(t_unique, ma_24, label="ma_24h")
        # ax2.plot(t_unique, ma_72, label="ma_72h")
        ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="0")
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Funding Rate * Volume trend")
        ax.legend()
        return fig, ax

    def plot_market_trend_ratio(self, fig=None, ax=None, label=None):
        t_list = []
        fr_list = []
        vol_list = []
        for _, df in self.trade_data.items():
            t_list.append(df["time"].values)
            fr_list.append(df["funding_rate"].values)
            vol_list.append(df["vol"].values)
        t = np.concatenate(t_list)
        fr = np.concatenate(fr_list)
        vol = np.concatenate(vol_list)
        fr_vol = fr * vol
        t_unique, t_idx = np.unique(t, return_inverse=True)
        fr_vol_pos = np.zeros_like(t_unique, dtype=np.float64)
        fr_vol_neg = np.zeros_like(t_unique, dtype=np.float64)
        for i in range(len(t_unique)):
            fr_vol_pos[i] = np.sum(fr_vol[t_idx == i][fr_vol[t_idx == i] > 0])
            fr_vol_neg[i] = np.sum(fr_vol[t_idx == i][fr_vol[t_idx == i] < 0])
        fr_vol_pos_ratio = fr_vol_pos / (fr_vol_pos - fr_vol_neg)
        fr_vol_neg_ratio = fr_vol_neg / (fr_vol_pos - fr_vol_neg)
        # get above 0.5 part
        fr_vol_pos_major = np.clip(fr_vol_pos_ratio, 0.5, 1)
        fr_vol_neg_major = np.clip(fr_vol_pos_ratio, 0, 0.5)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t_unique, fr_vol_pos_major, label="pos", alpha=0.5, color="red")
        ax.plot(t_unique, fr_vol_neg_major, label="neg", alpha=0.5, color="green")
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="0.5")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Funding Rate * Volume trend ratio")
        ax.legend()
        return fig, ax

    def plot_pnl_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(data=self.trade_record, x="time", y="pnl", ax=ax, label=label)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="0")
        ax.set_title("PnL curve")
        return fig, ax

    def plot_pnl_hist(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(data=self.trade_record, x="pnl", bins=40, ax=ax, label=label)
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

    def plot_pnl_fee_relation(self, fig=None, ax=None, label=None):
        """Plot PnL/fee vs signal std"""
        # get data
        x_std = []
        y = []
        for symbol, cur_fee in self.symbol_fee.items():
            cur_pnl = self.symbol_pnl[symbol]  # Fee not included
            y.append((cur_pnl - cur_fee) / cur_fee)
            pred = self.trade_data[symbol]["signal"].values
            x_std.append(np.std(pred))
        sns.scatterplot(x=x_std, y=y, ax=ax, label=label)
        # linear fit x, y
        x_min, x_max = np.min(x_std), np.max(x_std)
        x_std = np.array(x_std).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_std, y)
        x_range = x_max - x_min
        xx = np.linspace(x_min - x_range * 0.1, x_max + x_range * 0.1, 100).reshape(
            -1, 1
        )
        yy = model.predict(xx)
        y_pred = model.predict(x_std)
        r2 = r2_score(y, y_pred)
        corr = np.corrcoef(x_std.reshape(-1), y.reshape(-1))[0, 1]
        # plot
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(xx, yy, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Signal std")
        ax.set_ylabel("PnL / Fee")
        ax.set_title(f"R2: {r2:.3f}, Corr: {corr:.3f}")
        return fig, ax

    def plot_pnl_rate_vs_std(self, fig=None, ax=None, label=None):
        """Plot PnL/mean_cap vs signal std"""
        # get data
        x_std = []
        y = []
        n_ticks = self.trade_record.shape[0]
        for symbol, cur_fee in self.symbol_fee.items():
            cur_pnl = self.symbol_pnl[symbol]  # Fee not included
            cur_pnl -= cur_fee
            cur_cap_integral = 0
            for d in self.symbol_data[symbol]:
                cur_cap_integral += d[1]
            cur_cap_mean = cur_cap_integral / n_ticks

    def plot_vol_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(
            data=self.trade_record,
            x="time",
            y="vol_buy",
            ax=ax,
            label=f"{label}-buy" if label is not None else None,
        )
        sns.lineplot(
            data=self.trade_record,
            x="time",
            y="vol_sell",
            ax=ax,
            label=f"{label}-sell" if label is not None else None,
        )
        ax.set_title("vol curve")
        return fig, ax

    def plot_vol_hist(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        x1 = self.trade_record["vol_buy"].values
        x2 = self.trade_record["vol_sell"].values
        x = np.concatenate([x1, x2])
        sns.histplot(x=x, ax=ax, label=label)
        ax.set_title("vol hist")
        return fig, ax

    def plot_vol_hour_hist(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        t = self.trade_record["time"].values
        # hh = [datetime.fromtimestamp(tt).hour for tt in t]
        hh = pd.to_datetime(t).hour.values % 8
        x1 = self.trade_record["vol_buy"].values
        x2 = self.trade_record["vol_sell"].values
        hour_dict_buy = defaultdict(int)
        hour_dict_sell = defaultdict(int)
        for i in range(len(hh)):
            hour = hh[i]
            hour_dict_buy[hour] += x1[i]
            hour_dict_sell[hour] += x2[i]
        # barplot
        sns.barplot(
            x=list(hour_dict_buy.keys()),
            y=list(hour_dict_buy.values()),
            ax=ax,
            label="buy",
        )
        sns.barplot(
            x=list(hour_dict_sell.keys()),
            y=list(hour_dict_sell.values()),
            ax=ax,
            label="sell",
        )
        ax.set_title("vol hour hist")
        return fig, ax

    def plot_cap_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(
            data=self.trade_record, x="time", y="cap_free", ax=ax, label="cap_free"
        )
        # sns.lineplot(data=self.trade_record, x="time", y="cap_used", ax=ax, label="cap_used")
        ax.axhline(y=self.cap, color="r", linestyle="--", label="cap_total")
        ax.set_title("capital usage history")
        ax.legend()
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

    def __init__(
        self,
        trade_data,
        fee=10e-4,
        preprocess=True,
        mp_hash=True,
        sell_point=0,
        buy_point=0,
        vol_limit=0.02,
        trade_features=None,
    ):
        super().__init__(
            trade_data,
            fee=fee,
            preprocess=preprocess,
            mp_hash=mp_hash,
            trade_features=trade_features,
        )
        self.cap = None
        self.sell_point = sell_point
        self.buy_point = buy_point
        self.vol_lim = vol_limit
        self.hold_lim = 0.1
        self.tick_vol_lim = 0.05
        self.warmup_ticks = -1
        self.warmup_cap_ratio = 0.01

    def __copy__(self):
        new_self = TradeSimulatorSignalSimple(self.trade_data, preprocess=False)
        new_self.__dict__.update(self.__dict__)
        new_self.trade_record = None
        new_self.trade_done = False
        return new_self

    def set_sell_point(self, sell_point):
        self.sell_point = sell_point

    def set_buy_point(self, buy_point):
        self.buy_point = buy_point

    def trade(self, cap=None, show_progress=True):
        logger.debug("Simulating trading")
        if cap is None:
            if self.cap is None:
                logger.error("Can't simulate trade because cap is not set")
                return
        else:
            self.cap = cap
        cap_free = self.cap
        trade_record = {
            "time": [],
            "cap_free": [],
            "cap_used": [],
            "n_symbol": [],
            "pnl": [],
            "vol_buy": [],
            "vol_sell": [],
        }
        symbol_data = defaultdict(list)
        symbol_pnl = defaultdict(float)
        symbol_fee = defaultdict(float)
        share_holds = defaultdict(float)
        share_caps = defaultdict(float)
        cum_pnl = 0
        if show_progress:
            tq_ticks = tqdm(self.ticks)
        else:
            tq_ticks = self.ticks
        for tick_id, tick in enumerate(tq_ticks):
            pnl = 0
            cur_buy = 0
            cur_sell = 0
            # check if crypto trading hour (00:00, 08:00, 16:00)
            if pd.to_datetime(tick).hour in [0, 8, 16]:
                for symbol, hold in share_holds.items():
                    cur = self.get_data(symbol, tick)
                    cur_pnl = hold * cur["price"] * cur["funding_rate"]
                    pnl += cur_pnl
                    symbol_pnl[symbol] += cur_pnl
            # sell shares with negative signal
            for symbol, hold in share_holds.items():
                cur = self.get_data(symbol, tick)
                if cur["signal"] > self.sell_point or hold == 0:
                    continue
                # free capital
                if hold <= cur["vol"] * self.vol_lim / cur["price"]:
                    reduced_hold = hold
                    reduced_cap = share_caps[symbol]
                    share_holds[symbol] = 0
                    share_caps[symbol] = 0
                else:
                    reduced_hold = cur["vol"] * self.vol_lim / cur["price"]
                    reduced_cap = share_caps[symbol] * reduced_hold / hold
                    share_holds[symbol] -= reduced_hold
                    share_caps[symbol] -= reduced_cap
                cur_fee = reduced_hold * cur["price"] * self.fee
                pnl -= cur_fee
                symbol_fee[symbol] += cur_fee
                cap_free += reduced_cap
                cur_sell += reduced_cap
                logger.debug(f"sell {reduced_hold} {symbol} at {cur['price']}")

            # buy top positive signals
            sig_sybs = []
            for symbol in self.symbols:
                cur = self.get_data(symbol, tick)
                sig_sybs.append((cur["signal"], symbol))
            sig_sybs = sorted(sig_sybs, reverse=True)
            for cur_sig, symbol in sig_sybs:
                if cap_free <= 0.01:
                    break
                if cur_sig <= self.buy_point:
                    break
                cur = self.get_data(symbol, tick)
                cap_free_tmp = cap_free
                if tick_id < self.warmup_ticks:
                    cap_free_tmp = min(cap_free, self.cap * self.warmup_cap_ratio)
                # buy
                cur_cap = min(
                    cap_free_tmp,
                    cur["vol"] * self.vol_lim,
                    self.cap * self.hold_lim - share_caps[symbol],
                )
                new_hold = cur_cap / cur["price"]
                share_holds[symbol] += new_hold
                share_caps[symbol] += cur_cap
                cur_fee = cur_cap * self.fee
                pnl -= cur_fee
                symbol_fee[symbol] += cur_fee
                cap_free -= cur_cap
                cur_buy += cur_cap
                logger.debug(f"buy {new_hold} {symbol} at {cur['price']}")
            # record
            trade_record["time"].append(tick)
            trade_record["cap_free"].append(cap_free)
            trade_record["cap_used"].append(self.cap - cap_free)
            trade_record["pnl"].append(pnl)
            trade_record["vol_buy"].append(cur_buy)
            trade_record["vol_sell"].append(-cur_sell)
            n_symbol = 0
            for symbol, hold in share_holds.items():
                if hold > 0:
                    n_symbol += 1
                    symbol_data[symbol].append(
                        (
                            tick,
                            share_caps[symbol],
                            hold,
                            symbol_pnl[symbol],
                            symbol_fee[symbol],
                        )
                    )  # if need to add extra data, must not change the existing order
            trade_record["n_symbol"].append(n_symbol)
            cum_pnl += pnl
            if show_progress:
                cum_pnl_rate = cum_pnl / self.cap * 100
                tq_ticks.set_description(f"cum_pnl_rate: {cum_pnl_rate:.2f}%")
        self.trade_record = pd.DataFrame(trade_record)
        self.symbol_data = symbol_data
        self.symbol_pnl = symbol_pnl
        self.symbol_fee = symbol_fee
        self.trade_done = True

    def plot_book_simple(self, save_dir, top_n=10, plot_shift=200):
        save_dir.mkdir(exist_ok=True)
        # PnL
        pnl_dir = save_dir / "pnl"
        pnl_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_pnl_cum_curve_rate(ax=ax)
        fig.savefig(pnl_dir / "pnl_cum_curve_rate.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_pnl_ma_curve_rate(ax=ax)
        fig.savefig(pnl_dir / "pnl_ma_curve_rate.png")
        fig, ax = plt.subplots(figsize=(18, 6))
        # symbol capital history
        symbol_dir = save_dir / "symbol"
        symbol_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(18, 6))
        self.plot_symbol_cap(ax=ax, top_n=top_n, plot_shift=plot_shift)
        fig.savefig(symbol_dir / f"cap_top{top_n}_history.png")

    def plot_book(self, save_dir):
        if not self.trade_done:
            logger.error("Trade not done yet, will not make plot book")
            return
        self.plot_book_base(save_dir)
        self.plot_book_simple(save_dir)
        # close all figures
        plt.close("all")

    def plot_symbol_cap(self, fig=None, ax=None, top_n=10, plot_shift=200):
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))
        sd = self.symbol_data
        sb_max_vol = {}
        for symbol, data in sd.items():
            sb_max_vol[symbol] = max([d[1] for d in data])
        top_symbols = sorted(sb_max_vol.items(), key=lambda x: x[1], reverse=True)
        # plot history
        for i, (symbol, _) in enumerate(top_symbols[:top_n]):
            data = sd[symbol]
            cur_shift = plot_shift * i
            ax.plot(
                [d[0] for d in data], [d[1] + cur_shift for d in data], label=symbol
            )
        ax.legend()

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


class TradeSimulatorSignalGeneral(TradeSimulatorSignalSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal = None
        self.strategy = None
        self.strategy_ready = False

    def __copy__(self):
        new_self = TradeSimulatorSignalGeneral(self.trade_data, preprocess=False)
        new_self.__dict__.update(self.__dict__)
        new_self.trade_record = None
        new_self.trade_done = False
        if self.strategy_ready:
            new_self.set_strategy(self.strategy_class, **(self.strategy_kwargs))
        return new_self

    def set_strategy(self, strategy, **kwargs):
        strategy_name = strategy.__name__
        logger.debug(f"Use strategy: {strategy_name}")
        self.strategy_class = strategy
        self.strategy_kwargs = kwargs
        self.strategy = strategy(self, **kwargs)
        self.strategy_ready = True

    def trade(self):
        if self.cap is None:
            logger.error("Can't simulate trade because cap is not set")
            return
        if not self.strategy_ready:
            logger.error("Can't simulate trade because strategy is not set")
            return
        self.strategy.run()
        self.trade_done = True


class TradeSimulatorSignalGeneralNegtive(TradeSimulatorSignalGeneral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal = None
        self.strategy = None
        self.strategy_ready = False
    
    def hash_trade_data_worker(self, symbol):
        hash_dict = {}
        df = self.trade_data[symbol]
        for i, row in df.iterrows():
            tick = pd.to_datetime(row["time"])
            hash_dict[tick] = {
                "price": row["price"],
                "funding_rate": row["funding_rate"],
                "vol": row["vol"],
                "signal": row["signal"],
                "amount": row["amount"],
                "interest": row["interest"],
            }
        return symbol, hash_dict



# Naive attempt, not working well, to be removed
class TradeSimulatorSignalPeriodic(TradeSimulatorSignalSimple):
    def __init__(self, *args, period=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.tick_vol_lim = 1 / period

    def trade(self, cap, show_progress=True):
        self.cap = cap
        cap_free = cap
        trade_record = {
            "time": [],
            "cap_free": [],
            "cap_used": [],
            "n_symbol": [],
            "pnl": [],
            "vol_buy": [],
            "vol_sell": [],
        }
        cache = deque(maxlen=self.period)
        share_holds = defaultdict(float)
        share_caps = defaultdict(float)
        cum_pnl = 0
        if show_progress:
            tq_ticks = tqdm(self.ticks)
        else:
            tq_ticks = self.ticks
        for tick in tq_ticks:
            pnl = 0
            cur_buy = 0
            cur_sell = 0
            # check if crypto trading hour (00:00, 08:00, 16:00)
            if pd.to_datetime(tick).hour in [0, 8, 16]:
                for symbol, hold in share_holds.items():
                    cur = self.get_data(symbol, tick)
                    pnl += hold * cur["price"] * cur["funding_rate"]
            # plan sells
            sell_plan = {}
            if len(cache) >= self.period:
                sell_plan = cache.popleft()
            # rank to_buy list
            sig_sybs = []
            for symbol in self.symbols:
                cur = self.get_data(symbol, tick)
                if cur["signal"] <= self.buy_point:
                    continue
                sig_sybs.append((cur["signal"], symbol))
            # pre-free capital (fee to be determined later)
            for symbol, (hold, cost) in sell_plan.items():
                share_holds[symbol] -= hold
                share_caps[symbol] -= cost
                cap_free += cost
            # buy top signals
            tick_data = {}
            for _, symbol in sig_sybs:
                if cap_free <= 0.01:
                    break
                cur = self.get_data(symbol, tick)

                if self.cap * self.tick_vol_lim - cur_buy <= 0.01:
                    break
                else:
                    tick_lim = self.cap * self.tick_vol_lim - cur_buy
                buy_cap = min(
                    cap_free,
                    tick_lim,
                    cur["vol"] * self.vol_lim,
                    self.cap * self.hold_lim - share_caps[symbol],
                )
                # cancel out buy / sell if possible
                if symbol in sell_plan:
                    sell_hold, sell_cost = sell_plan[symbol]
                    if buy_cap >= sell_cost:
                        sell_plan[symbol] = None
                        buy_cap_new = buy_cap - sell_cost
                        hold_new = buy_cap_new / cur["price"]
                        share_holds[symbol] += sell_hold + hold_new
                        share_caps[symbol] += sell_cost + buy_cap_new
                        pnl -= buy_cap_new * self.fee
                        cap_free -= sell_cost + buy_cap_new
                        cur_buy += buy_cap_new
                        tick_data[symbol] = (
                            sell_hold + hold_new,
                            sell_cost + buy_cap_new,
                        )
                    else:  # no buy
                        sell_cost_new = sell_cost - buy_cap
                        sell_hold_new = sell_cost_new / cur["price"]
                        sell_plan[symbol] = (sell_hold_new, sell_cost_new)
                        share_caps[symbol] += buy_cap
                        share_holds[symbol] += buy_cap / cur["price"]
                        cap_free -= buy_cap
                        tick_data[symbol] = (buy_cap / cur["price"], buy_cap)
                else:
                    hold_new = buy_cap / cur["price"]
                    share_holds[symbol] += hold_new
                    share_caps[symbol] += buy_cap
                    cap_free -= buy_cap
                    cur_buy += buy_cap
                    tick_data[symbol] = (hold_new, buy_cap)
            # sell fee
            for symbol in sell_plan:
                sell_data = sell_plan[symbol]
                if sell_data is None:
                    continue
                sell_hold, sell_cost = sell_data
                cur_sell += sell_cost
                pnl -= sell_cost * self.fee
            # record
            trade_record["time"].append(tick)
            trade_record["cap_free"].append(cap_free)
            trade_record["cap_used"].append(self.cap - cap_free)
            trade_record["pnl"].append(pnl)
            trade_record["vol_buy"].append(cur_buy)
            trade_record["vol_sell"].append(-cur_sell)
            n_symbol = 0
            for symbol, hold in share_holds.items():
                if hold > 0:
                    n_symbol += 1
            trade_record["n_symbol"].append(n_symbol)
            cum_pnl += pnl
            if show_progress:
                cum_pnl_rate = cum_pnl / self.cap * 100
                tq_ticks.set_description(f"cum_pnl_rate: {cum_pnl_rate:.2f}%")
            cache.append(tick_data)
        self.trade_record = pd.DataFrame(trade_record)
        self.trade_done = True
