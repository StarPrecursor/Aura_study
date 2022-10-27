import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# Constants
fr_data_dir = Path("/home/shared/coin/tardis_derivative_ticker/")
badSymbolList = ["ICPUSDT"]
date_min = datetime.datetime(2022, 1, 1)
date_max = datetime.datetime(2022, 9, 30)


class FundingRateProcessor:
    def __init__(self, date_start=date_min, date_end=date_max):
        self.date_start = date_start
        self.date_end = date_end

    def get_symbol_list(self, sort=True, logic="and"):
        all_symbols = set()
        for i, date in enumerate(
            pd.date_range(self.date_start, self.date_end, freq="D")
        ):
            date_str = date.strftime("%Y-%m-%d")
            cur_symbols = set()
            for f in (fr_data_dir / date_str).glob("*.csv"):
                if f.stem in badSymbolList:
                    continue
                cur_symbols.add(f.stem)
            if logic == "or":
                all_symbols |= cur_symbols
            elif logic == "and":
                if i == 0:
                    all_symbols = cur_symbols
                else:
                    all_symbols &= cur_symbols
            else:
                raise ValueError(f"Unknown logic: {logic}")
        if sort:
            all_symbols = sorted(all_symbols)
        return list(all_symbols)

    def get_symbol_data(
        self,
        symbol,
        features=["funding_timestamp", "funding_rate"],
        mp=True,
        use_8h=True,
        resample=None,
    ):
        if "funding_timestamp" not in features:
            features = ["funding_timestamp"] + features
        if mp:
            with ProcessPoolExecutor() as executor:
                futures = []
                for date in pd.date_range(self.date_start, self.date_end, freq="D"):
                    futures.append(
                        executor.submit(
                            self.get_symbol_data_single_day,
                            symbol,
                            date,
                            features,
                            resample,
                        )
                    )
            df_list = [f.result() for f in futures if f.result() is not None]
        else:
            df_list = []
            for date in pd.date_range(self.date_start, self.date_end, freq="D"):
                df_tmp = self.get_symbol_data_single_day(
                    symbol, date, features, resample
                )
                if df_tmp is not None:
                    df_list.append(df_tmp)
        if use_8h:
            for i, df_tmp in enumerate(df_list):
                df_list[i] = df_tmp.loc[
                    df_tmp["funding_timestamp"].diff() > 0, features
                ]
        df_out = pd.concat(df_list).reset_index(drop=True)
        df_out["symbol"] = symbol
        return df_out

    def get_symbol_data_single_day(
        self,
        symbol,
        date,
        features=["funding_timestamp", "funding_rate"],
        resample=None,
        bfill=True,
    ):
        features_load = features.copy()
        if "timestamp" not in features_load:
            features_load.append("timestamp")
        if "funding_timestamp" not in features_load:
            features_load.append("funding_timestamp")
        date_str = date.strftime("%Y-%m-%d")
        f = fr_data_dir / date_str / f"{symbol}.csv"
        if not f.exists():
            return None
        df_tmp = pd.read_csv(f, usecols=features)
        if resample is not None:
            df_tmp["time"] = pd.to_datetime(df_tmp["timestamp"], unit="us")
            df_tmp = df_tmp.resample(resample, on="time").mean().reset_index()
        if bfill:
            df_tmp = df_tmp.bfill()
        return df_tmp[features]


def merge_symbols_mp(func, symbols):
    with ProcessPoolExecutor() as executor:
        futures = []
        for symbol in symbols:
            futures.append(executor.submit(func, symbol))
        df_list = [f.result() for f in futures]
    df_out = pd.concat(df_list).reset_index(drop=True)
    return df_out