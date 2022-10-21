from pathlib import Path

from artool.ar_io.downloaders import Binance_Downloader

save_dir = Path("/home/yangzhe/data/binance")

bd = Binance_Downloader("um", "1d", save_dir)
bd.download_daily_klines("2022-01-01", "2022-01-01")
