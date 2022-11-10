import logging
from asyncio import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from artool.ar_io.downloaders import BinanceDownloader

logging.basicConfig(level=logging.INFO)

date_start = "2022-09-01"
date_end = "2022-11-08"
save_dir = Path("/home/yangzhe/data/binance")

for market in ["spot", "um"]:
    # Download spot
    bd = BinanceDownloader(market, "1m", save_dir)
    futures = []
    with ProcessPoolExecutor() as executor:
        for date_cur in pd.date_range(date_start, date_end):
            date_str = date_cur.strftime("%Y-%m-%d")
            futures.append(executor.submit(bd.download_daily_klines, date_str, date_str))
        for future in tqdm(futures):
            future.result()
    # get failed logs
    fail_logs = bd.fail_logs
    print("#### Failed logs:")
    for log in fail_logs:
        print(log)
    # write failed logs to file
    with open(f"failed_logs_{market}.log", "w") as f:
        f.write("#### Failed logs:")
        for log in fail_logs:
            f.write(log)
