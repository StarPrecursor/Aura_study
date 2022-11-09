# pip install tardis-dev
# requires Python >=3.6
import os
import sys
from datetime import datetime, timedelta
from tardis_dev import datasets, get_exchange_details
import logging
import dateutil.parser
import gzip
import shutil
import argparse

# comment out to disable debug logs
logging.basicConfig(level=logging.DEBUG)

# function used by default if not provided via options
def default_file_name(exchange, data_type, date, symbol, format):
    return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    # return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"
    return f"{date.strftime('%Y-%m-%d')}/{symbol}.{format}.gz"


class TradisDerivativeClient:
    def __init__(self, base_path, exchange):
        def getFutureExchangeFromSpot(exchange):
            if exchange == "binance":
                futures = "binance-futures"
            elif exchange == "okex":
                futures = "okex-swap"

            return futures

        self.m_mBasePath = base_path
        self.m_mDev_api_key = "TD.oH9mnwzoG-qraSEV.b6YRyThteVoDNoX.C0khgwjT27GrgEY.9DVl7eetrdspaj8.XLq9Omjf3MXVaMU.JO1j"
        self.m_mExchange = getFutureExchangeFromSpot(exchange)
        self.m_mspotExchange = exchange
        self.m_mDataTypes = ["derivative_ticker"]
        self.m_mGotSymbolsFromTardis = []
        self.m_mGotSpotSymbolsFromTardis = []

        self.Init()

    def Init(self):
        exchange_details = get_exchange_details(self.m_mExchange)
        self.m_mGotSymbolsFromTardis = exchange_details["datasets"]["symbols"]

        exchange_details = get_exchange_details(self.m_mspotExchange)
        self.m_mGotSpotSymbolsFromTardis = exchange_details["datasets"]["symbols"]
        # for symbol_item in self.m_mGotSpotSymbolsFromTardis:
        #    print(symbol_item)

    def GetSymbolByDay(self, day):
        symbols = []
        print(len(self.m_mGotSymbolsFromTardis), len(self.m_mGotSpotSymbolsFromTardis))
        # for symbol_item in self.m_mGotSymbolsFromTardis:
        # print(symbol_item['id'], 'future')
        # for spot_symbol_item in self.m_mGotSpotSymbolsFromTardis:
        #    print(spot_symbol_item['id'], 'spot')

        for symbol_item in self.m_mGotSymbolsFromTardis:
            # print(symbol_item['id'], 'future')
            day_since = datetime.strptime(
                symbol_item["availableSince"], "%Y-%m-%dT%H:%M:%S.000Z"
            )
            day_to = datetime.strptime(
                symbol_item["availableTo"], "%Y-%m-%dT%H:%M:%S.000Z"
            )

            if (
                "derivative_ticker" in symbol_item["dataTypes"]
                and day_since <= day < day_to
            ):
                if symbol_item["id"] not in ["PERPETUALS", "FUTURES"] and (
                    "_" not in symbol_item["id"]
                ):
                    for spot_symbol_item in self.m_mGotSpotSymbolsFromTardis:
                        # print(spot_symbol_item['id'], 'spot')
                        if exchange == "okex":
                            mappedFutureId = "-".join(symbol_item["id"].split("-")[:2])
                        else:
                            mappedFutureId = symbol_item["id"]
                        if mappedFutureId == spot_symbol_item["id"]:
                            spot_day_since = datetime.strptime(
                                spot_symbol_item["availableSince"],
                                "%Y-%m-%dT%H:%M:%S.000Z",
                            )
                            spot_day_to = datetime.strptime(
                                spot_symbol_item["availableTo"],
                                "%Y-%m-%dT%H:%M:%S.000Z",
                            )
                            if spot_day_since <= day < spot_day_to:
                                symbols.append(symbol_item["id"])

        print(day, len(symbols), "number of symbols")

        return symbols

    def Download(self, symbols, from_date, to_date):
        datasets.download(
            exchange=self.m_mExchange,
            data_types=self.m_mDataTypes,
            from_date=from_date,
            to_date=to_date,
            symbols=symbols,
            api_key=self.m_mDev_api_key,
            download_dir=self.m_mBasePath,
            get_filename=file_name_nested,
        )
        current_date = dateutil.parser.isoparse(from_date)
        for symbol in symbols:
            if symbol != 'PERPETUALS':
                file_name = f"{self.m_mBasePath}/" + file_name_nested(exchange="", data_type=[], symbol=symbol,
                                                                      date=current_date,
                                                                      format="csv")
                uncompress_file_name = file_name[:-3]
                with gzip.open(file_name, 'rb') as f_in:
                    with open(uncompress_file_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                f_in.close()
                f_out.close()
                os.remove(file_name)

    def Run(self, begin_date, end_date, symbols):
        day = datetime.strptime(begin_date, "%Y-%m-%d")
        day_end = datetime.strptime(end_date, "%Y-%m-%d")

        while day < day_end:
            # if len(symbols) == 0:
            symbols = self.GetSymbolByDay(day)

            f_curr_day = day.strftime("%Y-%m-%d")

            next_day = day + timedelta(days=1)
            f_next_day = next_day.strftime("%Y-%m-%d")
            self.Download(symbols, f_curr_day, f_next_day)
            day = next_day


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("from_date", type=str, help="start date, eg 2022-09-26 ")
    parser.add_argument("to_date", type=str, help="to date, eg 2022-09-26 ")
    parser.add_argument("exchange", type=str, help="exchange group, eg binance")
    parser.add_argument(
        "-v", "--verbosity", action="store_true", help=" output verbosity"
    )

    args = parser.parse_args()

    from_date = args.from_date
    to_date = args.to_date
    exchange = args.exchange

    #''''''
    symbols = []
    """
    if len(sys.argv) >= 3:
        from_date = sys.argv[1]
        to_date = sys.argv[2]
        for i in range(3, len(sys.argv)):
            symbols.append(sys.argv[i])
    else:
        print("Params issue: length of params is less then 3")
        exit(-1)
    """

    # base_path = os.path.join('/home/shared/', 'coin', "tardis_derivative_ticker", exchange)
    # base_path = os.path.join(".")
    base_path = os.path.join('/home/shared/', 'coin', "tardis_derivative_ticker")
    tardis_client = TradisDerivativeClient(base_path, exchange)
    tardis_client.Run(from_date, to_date, symbols)

    # python.exe.\get_tardis_derivative.py 2022-09-12 2022-09-14
