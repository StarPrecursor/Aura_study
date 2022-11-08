from functools import partial
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("artool")


class StrategyBase:
    def __init__(self, simu, show_progress=True):
        self.simu = simu
        self.reset_data()
        self.show_progress = show_progress
        self.cur_pnl = 0
        self.cur_buy = 0
        self.cur_sell = 0
        self.cum_pnl = 0
        self.cap_free = 0

    def reset_data(self):
        # Output
        self.trade_record = {
            "time": [],
            "cap_free": [],
            "cap_used": [],
            "n_symbol": [],
            "pnl": [],
            "vol_buy": [],
            "vol_sell": [],
        }
        self.symbol_data = defaultdict(list)
        self.symbol_pnl = defaultdict(float)
        self.symbol_fee = defaultdict(float)
        # Cache
        self.share_holds = defaultdict(float)
        self.share_caps = defaultdict(float)

    def pass_data(self):
        self.simu.trade_record = pd.DataFrame(self.trade_record)
        self.simu.symbol_data = deepcopy(self.symbol_data)
        self.simu.symbol_pnl = deepcopy(self.symbol_pnl)
        self.simu.symbol_fee = deepcopy(self.symbol_fee)

    def set_ticks(self):
        if self.show_progress:
            self.ticks = tqdm(self.simu.ticks)
        else:
            self.ticks = self.simu.ticks

    def update_ticks(self):
        if self.show_progress:
            cum_pnl_rate = self.cum_pnl / self.simu.cap * 100
            self.ticks.set_description(f"cum_pnl_rate: {cum_pnl_rate:.2f}%")

    def fund(self, tick):
        simu = self.simu
        # check if crypto trading hour (00:00, 08:00, 16:00)
        if pd.to_datetime(tick).hour in [0, 8, 16]:
            for symbol, hold in self.share_holds.items():
                cur = simu.get_data(symbol, tick)
                cur_profit = hold * cur["price"] * cur["funding_rate"]
                self.cur_pnl += cur_profit
                self.symbol_pnl[symbol] += cur_profit

    def update(self, tick):
        simu = self.simu
        # record
        self.trade_record["time"].append(tick)
        self.trade_record["cap_free"].append(self.cap_free)
        self.trade_record["cap_used"].append(simu.cap - self.cap_free)
        self.trade_record["pnl"].append(self.cur_pnl)
        self.trade_record["vol_buy"].append(self.cur_buy)
        self.trade_record["vol_sell"].append(-self.cur_sell)
        n_symbol = 0
        for symbol, hold in self.share_holds.items():
            if hold > 0:
                n_symbol += 1
                self.symbol_data[symbol].append(
                    (
                        tick,
                        self.share_caps[symbol],
                        hold,
                        self.symbol_pnl[symbol],
                        self.symbol_fee[symbol],
                    )
                )  # if need to add extra data, must not change the existing order
        self.trade_record["n_symbol"].append(n_symbol)

    def get_ranked_signal(self, tick, simu):
        sig_sybs = []
        for symbol in simu.symbols:
            cur = simu.get_data(symbol, tick)
            sig_sybs.append((cur["signal"], symbol))
        sig_sybs = sorted(sig_sybs, reverse=True)
        return sig_sybs

    def add_hold(self, symbol, cur_cap, cur_price):
        # update hold
        new_hold = cur_cap / cur_price
        self.share_holds[symbol] += new_hold
        self.share_caps[symbol] += cur_cap
        cur_fee = cur_cap * self.simu.fee
        self.cur_pnl -= cur_fee
        self.symbol_fee[symbol] += cur_fee
        self.cap_free -= cur_cap
        self.cur_buy += cur_cap
        logger.debug(f"buy {new_hold} {symbol} at {cur_price}")


class ChampagneTower(StrategyBase):
    def __init__(self, simu, show_progress=True):
        super().__init__(simu, show_progress=show_progress)

    def sell(self, tick):
        """Sells shares with negative signal."""
        simu = self.simu
        for symbol, hold in self.share_holds.items():
            cur = simu.get_data(symbol, tick)
            if cur["signal"] > simu.sell_point or hold == 0:
                continue
            # free capital
            if hold <= cur["vol"] * simu.vol_lim / cur["price"]:
                reduced_hold = hold
                reduced_cap = self.share_caps[symbol]
                self.share_holds[symbol] = 0
                self.share_caps[symbol] = 0
            else:
                reduced_hold = cur["vol"] * simu.vol_lim / cur["price"]
                reduced_cap = self.share_caps[symbol] * reduced_hold / hold
                self.share_holds[symbol] -= reduced_hold
                self.share_caps[symbol] -= reduced_cap
            cur_fee = reduced_hold * cur["price"] * simu.fee
            self.cur_pnl -= cur_fee
            self.symbol_fee[symbol] += cur_fee
            self.cap_free += reduced_cap
            self.cur_sell += reduced_cap
            logger.debug(f"sell {reduced_hold} {symbol} at {cur['price']}")

    def buy(self, tick, tick_id):
        simu = self.simu
        # buy top positive signals
        sig_sybs = self.get_ranked_signal(tick, simu)
        for cur_sig, symbol in sig_sybs:
            if self.cap_free <= 0.01:
                break
            if cur_sig <= simu.buy_point:
                break
            # determine assigned share
            cur = simu.get_data(symbol, tick)
            cap_free_tmp = self.cap_free
            if tick_id < simu.warmup_ticks:
                cap_free_tmp = min(cap_free_tmp, simu.cap * simu.warmup_cap_ratio)
            cur_cap = min(
                cap_free_tmp,
                cur["vol"] * simu.vol_lim,
                simu.cap * simu.hold_lim - self.share_caps[symbol],
            )
            self.add_hold(symbol, cur_cap, cur["price"])

    def run(self, pass_data=True):
        simu = self.simu
        self.cum_pnl = 0
        self.cap_free = simu.cap
        self.set_ticks()
        for tick_id, tick in enumerate(self.ticks):
            self.cur_pnl = 0
            self.cur_buy = 0
            self.cur_sell = 0
            # process
            self.fund(tick)
            self.sell(tick)
            self.buy(tick, tick_id)
            # updates
            self.update(tick)
            self.cum_pnl += self.cur_pnl
            self.update_ticks()
        if pass_data:
            self.pass_data()


class ChampagneTowerHourlyWeighted(ChampagneTower):
    def __init__(self, simu, show_progress=True):
        super().__init__(simu, show_progress=show_progress)

    def get_hour_weight(self, tick):
        hour = pd.to_datetime(tick).hour % 8
        return np.power(2, -(7 - float(hour))).astype(
            float
        )  # Integers to negative integer powers are not allowed

    def sell(self, tick):
        """Sells shares with negative signal."""
        simu = self.simu
        h_wt = self.get_hour_weight(tick)
        for symbol, hold in self.share_holds.items():
            cur = simu.get_data(symbol, tick)
            if cur["signal"] > simu.sell_point or hold == 0:
                continue
                # free capital
            if hold <= cur["vol"] * simu.vol_lim / cur["price"]:
                reduced_hold = hold
                reduced_cap = self.share_caps[symbol]
            else:
                reduced_hold = cur["vol"] * simu.vol_lim / cur["price"]
                reduced_cap = self.share_caps[symbol] * reduced_hold / hold
            # perform less changes for hours that far from trading hour
            reduced_hold *= h_wt
            reduced_cap *= h_wt
            # updates
            self.share_holds[symbol] -= reduced_hold
            self.share_caps[symbol] -= reduced_cap
            cur_fee = reduced_hold * cur["price"] * simu.fee
            self.cur_pnl -= cur_fee
            self.symbol_fee[symbol] += cur_fee
            self.cap_free += reduced_cap
            self.cur_sell += reduced_cap
            logger.debug(f"sell {reduced_hold} {symbol} at {cur['price']}")

    def buy(self, tick, tick_id):
        simu = self.simu
        # buy top positive signals
        sig_sybs = self.get_ranked_signal(tick, simu)
        h_wt = self.get_hour_weight(tick)
        for cur_sig, symbol in sig_sybs:
            if self.cap_free <= 0.01:
                break
            if cur_sig <= simu.buy_point:
                break
            # determine assigned share
            cur = simu.get_data(symbol, tick)
            cap_free_tmp = self.cap_free * h_wt  # apply hour weight
            if tick_id < simu.warmup_ticks:
                cap_free_tmp = min(cap_free_tmp, simu.cap * simu.warmup_cap_ratio)
            cur_cap = min(
                cap_free_tmp,
                cur["vol"] * simu.vol_lim,
                simu.cap * simu.hold_lim - self.share_caps[symbol],
            )
            self.add_hold(symbol, cur_cap, cur["price"])


def relu(x):
    return np.maximum(x, 0)


def linear(x, a=1.0, b=0):
    return a * x + b


def sigmoid(x, c=1.0):
    return 1 / (1 + np.exp(-x / c))


class WeightedTarget(ChampagneTower):
    def __init__(self, simu, show_progress=True, activation="relu", atv_params={}):
        super().__init__(simu, show_progress)
        self.symbol_wt = defaultdict(float)
        self.set_activation(activation, atv_params)

    def set_activation(self, activation: Union[Callable, str], atv_params):
        if isinstance(activation, str):
            if activation == "relu":
                self.atv = relu
            if activation == "linear":
                self.atv = partial(linear, **atv_params)
            else:
                raise ValueError(f"activation {activation} not supported")
        else:
            self.atv = activation

    def buy(self, tick, tick_id=None):  # tick_id is kept for compatibility
        simu = self.simu
        # plan symbol shares
        total_wt = 0
        for symbol in simu.symbols:
            cur = simu.get_data(symbol, tick)
            cur_wt = relu(self.atv(cur["signal"] - simu.buy_point))  # type: ignore
            self.symbol_wt[symbol] = cur_wt
            total_wt += cur_wt
        if total_wt == 0:
            logger.debug(f"No positive signal at {tick}, skip buying")
            return
        # normalize
        for symbol in simu.symbols:
            self.symbol_wt[symbol] /= total_wt
        # buy missing share from top to bottom
        sig_sybs = self.get_ranked_signal(tick, simu)
        for cur_sig, symbol in sig_sybs:
            if self.cap_free <= 0.01:
                break
            if cur_sig <= simu.buy_point:
                break
            # determine assigned share
            cur = simu.get_data(symbol, tick)
            cap_free_plan = self.cap_free * self.symbol_wt[symbol]
            if self.share_caps[symbol] >= cap_free_plan:
                continue
            cur_cap = min(
                cap_free_plan,
                cur["vol"] * simu.vol_lim,
                simu.cap * simu.hold_lim - self.share_caps[symbol],
            )
            # update hold
            self.add_hold(symbol, cur_cap, cur["price"])


class DynamicShare(StrategyBase):
    def __init__(self, simu, show_progress=True):
        super().__init__(simu, show_progress)
        self.symbol_wt = defaultdict(float)
        self.symbol_wt_norm = defaultdict(float)

    def update_share_wt(self, tick):
        simu = self.simu
        alpha = 1 - np.exp(-np.log(2) / 72)
        # update new weight
        total_wt = 0
        for symbol in simu.symbols:
            cur = simu.get_data(symbol, tick)
            cur_wt = sigmoid(
                cur["signal"] - simu.buy_point,
                (simu.buy_point - simu.sell_point) * 0.01,
            )
            # cur_wt = sigmoid(cur["signal"] - simu.buy_point, 0.0001 * 0.01)

            # cur_wt = relu(cur["signal"] - simu.buy_point)

            self.symbol_wt[symbol] = cur_wt * alpha + self.symbol_wt[symbol] * (
                1 - alpha
            )
            # self.symbol_wt[symbol] = cur_wt

            total_wt += self.symbol_wt[symbol]
        if total_wt == 0:
            logger.debug(f"No positive signal at {tick}, skip buying")
            return
        # normalize
        for symbol in simu.symbols:
            self.symbol_wt_norm[symbol] = self.symbol_wt[symbol] / total_wt

    def sell(self, tick):
        simu = self.simu
        for symbol, hold in self.share_holds.items():
            if hold == 0:
                continue  # can't sell
            cur = simu.get_data(symbol, tick)
            pre_cap = self.share_caps[symbol]
            nxt_cap = simu.cap * self.symbol_wt_norm[symbol]
            if nxt_cap >= pre_cap:  # need buy, not sell
                continue
            reduced_hold = min(
                hold * (pre_cap - nxt_cap) / pre_cap,  # as planned
                cur["vol"] * simu.vol_lim / cur["price"],  # market allowed
            )
            reduced_cap = pre_cap * reduced_hold / hold

            if reduced_cap < simu.cap * 0.005:
                continue

            # updates
            self.share_holds[symbol] -= reduced_hold
            self.share_caps[symbol] -= reduced_cap
            cur_fee = reduced_hold * cur["price"] * simu.fee
            self.cur_pnl -= cur_fee
            self.symbol_fee[symbol] += cur_fee
            self.cap_free += reduced_cap
            self.cur_sell += reduced_cap
            logger.debug(f"sell {reduced_hold} {symbol} at {cur['price']}")

    def buy(self, tick, tick_id=None):
        simu = self.simu
        # first fulfill top positive signals
        sig_sybs = self.get_ranked_signal(tick, simu)
        for cur_sig, symbol in sig_sybs:
            if self.cap_free <= 0.01:
                break
            # determine assigned share
            cur = simu.get_data(symbol, tick)
            pre_cap = self.share_caps[symbol]
            nxt_cap = simu.cap * self.symbol_wt_norm[symbol]
            if nxt_cap >= simu.cap * simu.hold_lim:
                nxt_cap = simu.cap * simu.hold_lim
            if nxt_cap <= pre_cap:  # need sell, not buy
                continue
            increased_cap = min(
                nxt_cap - pre_cap,  # as planned
                cur["vol"] * simu.vol_lim,  # allowed
                self.cap_free,
            )

            if increased_cap < simu.cap * 0.005:
                continue

            # update
            self.add_hold(symbol, increased_cap, cur["price"])

    def run(self, pass_data=True):
        simu = self.simu
        self.cum_pnl = 0
        self.cap_free = simu.cap
        self.set_ticks()
        for tick_id, tick in enumerate(self.ticks):
            self.cur_pnl = 0
            self.cur_buy = 0
            self.cur_sell = 0
            # process
            self.fund(tick)
            self.update_share_wt(tick)
            self.sell(tick)
            self.buy(tick, tick_id)
            # updates
            self.update(tick)
            self.cum_pnl += self.cur_pnl
            self.update_ticks()
        if pass_data:
            self.pass_data()
