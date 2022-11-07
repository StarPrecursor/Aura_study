import logging
from collections import defaultdict
from copy import deepcopy
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
                cap_free_tmp = min(self.cap_free, simu.cap * simu.warmup_cap_ratio)
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


class WeightedTarget(ChampagneTower):
    def __init__(self, simu, show_progress=True):
        super().__init__(simu, show_progress)
        self.symbol_wt = defaultdict(float)

    def buy(self, tick, tick_id=None):  # tick_id is kept for compatibility
        simu = self.simu
        # plan symbol shares
        total_wt = 0
        for symbol in simu.symbols:
            cur = simu.get_data(symbol, tick)
            cur_wt = max(cur["signal"] - simu.buy_point, 0)
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


def champagne_tower(simu, show_progress=True):
    """Fill strong signal volume first"""
    # Data out
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
    # Cache
    share_holds = defaultdict(float)
    share_caps = defaultdict(float)
    # Simulation
    cum_pnl = 0
    cap_free = simu.cap
    ticks = simu.ticks
    if show_progress:
        ticks = tqdm(ticks)
    for tick_id, tick in enumerate(ticks):
        pnl = 0
        cur_buy = 0
        cur_sell = 0
        # check if crypto trading hour (00:00, 08:00, 16:00)
        if pd.to_datetime(tick).hour in [0, 8, 16]:
            for symbol, hold in share_holds.items():
                cur = simu.get_data(symbol, tick)
                cur_pnl = hold * cur["price"] * cur["funding_rate"]
                pnl += cur_pnl
                symbol_pnl[symbol] += cur_pnl
        # sell shares with negative signal
        for symbol, hold in share_holds.items():
            cur = simu.get_data(symbol, tick)
            if cur["signal"] > simu.sell_point or hold == 0:
                continue
            # free capital
            if hold <= cur["vol"] * simu.vol_lim / cur["price"]:
                reduced_hold = hold
                reduced_cap = share_caps[symbol]
                share_holds[symbol] = 0
                share_caps[symbol] = 0
            else:
                reduced_hold = cur["vol"] * simu.vol_lim / cur["price"]
                reduced_cap = share_caps[symbol] * reduced_hold / hold
                share_holds[symbol] -= reduced_hold
                share_caps[symbol] -= reduced_cap
            cur_fee = reduced_hold * cur["price"] * simu.fee
            pnl -= cur_fee
            symbol_fee[symbol] += cur_fee
            cap_free += reduced_cap
            cur_sell += reduced_cap
            logger.debug(f"sell {reduced_hold} {symbol} at {cur['price']}")
        # buy top positive signals
        sig_sybs = []
        for symbol in simu.symbols:
            cur = simu.get_data(symbol, tick)
            sig_sybs.append((cur["signal"], symbol))
        sig_sybs = sorted(sig_sybs, reverse=True)
        for cur_sig, symbol in sig_sybs:
            if cap_free <= 0.01:
                break
            if cur_sig <= simu.buy_point:
                break
            cur = simu.get_data(symbol, tick)
            cap_free_tmp = cap_free
            if tick_id < simu.warmup_ticks:
                cap_free_tmp = min(cap_free, simu.cap * simu.warmup_cap_ratio)
            # buy
            cur_cap = min(
                cap_free_tmp,
                cur["vol"] * simu.vol_lim,
                simu.cap * simu.hold_lim - share_caps[symbol],
            )
            new_hold = cur_cap / cur["price"]
            share_holds[symbol] += new_hold
            share_caps[symbol] += cur_cap
            cur_fee = cur_cap * simu.fee
            pnl -= cur_fee
            symbol_fee[symbol] += cur_fee
            cap_free -= cur_cap
            cur_buy += cur_cap
            logger.debug(f"buy {new_hold} {symbol} at {cur['price']}")
        # record
        trade_record["time"].append(tick)
        trade_record["cap_free"].append(cap_free)
        trade_record["cap_used"].append(simu.cap - cap_free)
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
            cum_pnl_rate = cum_pnl / simu.cap * 100
            ticks.set_description(f"cum_pnl_rate: {cum_pnl_rate:.2f}%")
    # Data out
    return {
        "trade_record": trade_record,
        "symbol_data": symbol_data,
        "symbol_pnl": symbol_pnl,
        "symbol_fee": symbol_fee,
    }
