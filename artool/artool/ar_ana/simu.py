import logging
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import pandas as pd
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


class TradeSimulatorSignal(TradeSimulatorBase):
    def __init__(
        self, trade_data: dict, fee: float = 10e-4, preprocess=True, mp_hash=True
    ):
        super().__init__()
        self.trade_data = trade_data
        self.trade_hash = None
        self.symbols = list(trade_data.keys())
        if preprocess:
            self.hash_trade_data(mp=mp_hash)
            self.get_ticks()
        self.fee = fee
        self.trade_record = None
        self.trade_done = False

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
        self.plot_vol_hist(ax=ax)
        fig.savefig(save_dir / "vol_hist.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_n_symbol_curve(ax=ax)
        fig.savefig(save_dir / "n_symbol_curve.png")

    def plot_book(self, save_dir):
        self.plot_book_base(save_dir)
        # close all figures
        plt.close("all")

    def plot_pnl_curve(self, fig=None, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(data=self.trade_record, x="time", y="pnl", ax=ax, label=label)
        ax.set_title("PnL curve")
        return fig, ax

    def plot_pnl_hist(self, fig=None, ax=None, label=None):
        print("#### PnL hist")
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
        print("#### vol hist")
        if ax is None:
            fig, ax = plt.subplots()
        x1 = self.trade_record["vol_buy"].values
        x2 = self.trade_record["vol_sell"].values
        x = np.concatenate([x1, x2])
        sns.histplot(x=x, ax=ax, label=label)
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

    def __init__(
        self,
        trade_data,
        fee=10e-4,
        preprocess=True,
        mp_hash=True,
        sell_point=0,
        buy_point=0,
        vol_limit=0.02,
    ):
        super().__init__(trade_data, fee=fee, preprocess=preprocess, mp_hash=mp_hash)
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

    def trade(self, cap, show_progress=True):
        logger.debug("Simulating trading")
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
        symbol_data = defaultdict(list)
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
                    pnl += hold * cur["price"] * cur["funding_rate"]
            # sell shares with negative signal
            for symbol, hold in share_holds.items():
                cur = self.get_data(symbol, tick)
                if cur["signal"] > self.sell_point or hold == 0:
                    continue
                # free capital
                reduced_cap = min(share_caps[symbol], cur["vol"] * self.vol_lim)
                reduced_hold = reduced_cap / cur["price"]
                share_holds[symbol] -= reduced_hold
                share_caps[symbol] -= reduced_cap
                pnl -= reduced_cap * self.fee
                cap_free += reduced_cap
                cur_sell += reduced_cap
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
                pnl -= cur_cap * self.fee
                cap_free -= cur_cap
                cur_buy += cur_cap
                logger.debug(f"buy {new_hold} {symbol} at {cur['price']}")
            # record
            trade_record["time"].append(tick)
            trade_record["cap_free"].append(cap_free)
            trade_record["cap_used"].append(cap - cap_free)
            trade_record["pnl"].append(pnl)
            trade_record["vol_buy"].append(cur_buy)
            trade_record["vol_sell"].append(-cur_sell)
            n_symbol = 0
            for symbol, hold in share_holds.items():
                if hold > 0:
                    n_symbol += 1
                    symbol_data[symbol].append((tick, share_caps[symbol], hold))
            trade_record["n_symbol"].append(n_symbol)
            cum_pnl += pnl
            if show_progress:
                cum_pnl_rate = cum_pnl / self.cap * 100
                tq_ticks.set_description(f"cum_pnl_rate: {cum_pnl_rate:.2f}%")
        self.trade_record = pd.DataFrame(trade_record)
        self.symbol_data = symbol_data
        self.trade_done = True

    def plot_book_simple(self, save_dir, top_n=10, plot_shift=200):
        save_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_cum_curve_rate(ax=ax)
        fig.savefig(save_dir / "pnl_cum_curve_rate.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_pnl_ma_curve_rate(ax=ax)
        fig.savefig(save_dir / "pnl_ma_curve_rate.png")
        fig, ax = plt.subplots(figsize=(18, 6))
        # symbol capital history
        self.plot_symbol_cap(ax=ax, top_n=top_n, plot_shift=plot_shift)
        plot_dir = save_dir / "symbols"
        plot_dir.mkdir(exist_ok=True)
        fig.savefig(plot_dir / f"cap_top{top_n}_history.png")

    def plot_book(self, save_dir):
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
            ax.plot([d[0] for d in data], [d[1] + cur_shift for d in data], label=symbol)
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


class TradeSimulatorSignalRank(TradeSimulatorSignal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trade_record = None
        self.trade_done = False

    def trade(self, cap, show_progress=True):
        logger.debug("Simulating trading")
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


# Not working well
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
