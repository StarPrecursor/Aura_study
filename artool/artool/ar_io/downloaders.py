import datetime
import hashlib
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# from tqdm import tqdm

# set logging level
# logging.basicConfig(level=logging.INFO)


class Binance_Downloader:
    def __init__(self, trading_type, interval, folder):
        self.trading_type = trading_type
        self.interval = interval
        self.folder = str(folder)
        self.symbols = self.get_all_symbols()
        self.trading_type_path = f"data/futures/{trading_type}"
        self.fail_logs = []
        logging.info(f"Found {len(self.symbols)} symbols")
        logging.info(f"Trading type: {trading_type}")
        logging.info(f"Interval: {interval}")
        logging.info(f"Donwload folder: {folder}")

    def download_daily_klines(self, start_date, end_date, checksum: bool = True):
        # convert string to date
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        for i, symbol in enumerate(self.symbols):
            logging.info(f"Downloading {symbol} ({i+1}/{len(self.symbols)})")
            for date in pd.date_range(start_date, end_date):
                path = self.get_path("klines", "daily", symbol)
                date_str = date.strftime('%Y-%m-%d')
                file_name = (
                    f"{symbol.upper()}-{self.interval}-{date_str}.zip"
                )
                flag = self.download_file(path, file_name, checksum=checksum)
                if not flag:
                    self.fail_logs.append(file_name.split(".")[0])

    def download_file(
        self, base_path, file_name, n_attempts=5, checksum=True, unzip=True
    ):
        logging.info(f"Downloading {file_name}")
        base_path = Path(base_path)
        download_path = base_path / file_name
        save_dir = Path(f"{self.folder}/{base_path}")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name
        for attempt_id in range(n_attempts):
            if attempt_id > 0:
                logging.info(f"Retrying {attempt_id} download {file_name}")
            # try download
            is_success = True
            if not save_path.exists():
                try:
                    download_url = self.get_download_url(download_path)
                    dl_file = urllib.request.urlopen(download_url)
                    length = dl_file.getheader("content-length")
                    length = int(length)
                    blocksize = max(4096, length // 100)
                    # save to disc
                    with open(save_path, "wb") as out_file:
                        dl_progress = 0
                        while True:
                            buf = dl_file.read(blocksize)
                            if not buf:
                                break
                            dl_progress += len(buf)
                            out_file.write(buf)
                    logging.info(f"Downloaded {file_name}")
                except urllib.error.HTTPError:
                    logging.warn(f"Not found: {file_name}")
                    return True
                else:
                    # attempt fail
                    # logging.warn(f"Failed to download: {download_path}")
                    continue
            # check sum
            if checksum:
                try:
                    checksum_file = base_path / f"{file_name}.CHECKSUM"
                    checksum_url = self.get_download_url(checksum_file)
                    cs_file = urllib.request.urlopen(checksum_url)
                    # get hash
                    cs_sha256 = cs_file.read().decode("utf-8").split(" ")[0]
                    # get sha256sum

                    sha256 = hashlib.sha256()
                    with open(save_path, "rb") as f:
                        while True:
                            data = f.read(65536)
                            if not data:
                                break
                            sha256.update(data)
                    assert cs_sha256 == sha256.hexdigest()
                    logging.info(f"Checksum passed: {file_name}")
                except:
                    logging.error(f"Failed to validate CHECKSUM for {file_name}")
                    # delete file if exists
                    if save_path.exists():
                        save_path.unlink()
                    continue
            if unzip:
                try:
                    with zipfile.ZipFile(save_path, "r") as zip_ref:
                        zip_ref.extractall(save_dir)
                    logging.info(f"Unzipped {file_name}")
                except:
                    logging.error(f"Failed to unzip {file_name}")
                    continue
                # delete file if exists
                if save_path.exists():
                    save_path.unlink()
            if is_success:
                logging.info(f"Downloaded {file_name}")
                return True
        return False

    def get_all_symbols(self):
        if self.trading_type == "um":
            response = urllib.request.urlopen(
                "https://fapi.binance.com/fapi/v1/exchangeInfo"
            ).read()
        elif self.trading_type == "cm":
            response = urllib.request.urlopen(
                "https://dapi.binance.com/dapi/v1/exchangeInfo"
            ).read()
        else:
            response = urllib.request.urlopen(
                "https://api.binance.com/api/v3/exchangeInfo"
            ).read()
        return list(
            map(lambda symbol: symbol["symbol"], json.loads(response)["symbols"])
        )

    def get_download_url(self, file_url):
        return f"https://data.binance.vision/{file_url}"

    def get_path(self, market_data_type, time_period, symbol):
        return f"{self.trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{self.interval}/"

def get_column_names(market="binance", trading_type="um", data_type="klines"):
    if market == "binance":
        if trading_type == "um":
            if data_type == "klines":
                return [
                    "Open time",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Close time",
                    "Quote asset volume",
                    "Number of trades",
                    "Taker buy base asset volume",
                    "Taker buy quote asset volume",
                    "Ignore",
                ]
    logging.warn(f"Can't find column names for {market} {trading_type} {data_type}")
    return None