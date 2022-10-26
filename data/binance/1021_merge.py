from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from artool import ar_io

for market in ["spot", "um"]:
    data_dir = Path("/home/yangzhe/data/binance/data/futures/um/daily/klines/")
    if market == "spot":
        data_dir = Path("/home/yangzhe/data/binance/data/spot/daily/klines/")

    # get symbol list
    symbol_list = ar_io.helpers.fetch_names(data_dir, r"(.*)", sort=True, group_id=1)
    symbol_list = [s[0] for s in symbol_list]
    # print(symbol_list)

    # get dataframe colomns
    columns = ar_io.downloaders.get_column_names()
    print(columns)

    # merge dataframes
    for symbol in symbol_list:
        print(f"Processing symbol: {symbol}")
        symbol_dir = data_dir / symbol / "1m"
        with ProcessPoolExecutor() as executor:
            # get paths end with .csv
            files = ar_io.helpers.fetch_names(
                symbol_dir, r"(.*\.csv)", sort=True, group_id=0
            )
            paths = [symbol_dir / p[0] for p in files]
            # read csv files
            futures = []
            for p in paths:
                futures.append(executor.submit(pd.read_csv, p))
            df_list = []
            for future in futures:
                cur_df = future.result()
                cur_df.columns = columns
                df_list.append(cur_df)
            if len(df_list) == 0:
                continue
            df = pd.concat(df_list, axis=0, ignore_index=True)
            # set column name
            df.columns = columns
            # save to feather
            df.to_feather(symbol_dir / "merge.feather")
