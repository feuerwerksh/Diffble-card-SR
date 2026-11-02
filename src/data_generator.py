import argparse
import os
from datetime import datetime
from io import StringIO

import FinanceDataReader as fdr
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

PATH_RAW_DATA = os.path.join('data', 'raw')
PATH_DATA = os.path.join('data', 'daily')


def _out_path(name: str, start_date: str, end_date: str) -> str:
    tag = '{}_{}'.format(datetime.strptime(start_date, '%Y-%m-%d').strftime('%y%m%d'),
                         datetime.strptime(end_date, '%Y-%m-%d').strftime('%y%m%d'))
    return os.path.join(PATH_DATA, f'{name}_{tag}.csv')


def _wikipedia_tickers(url: str, candidates: list[str]) -> list[str]:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    for tbl in pd.read_html(StringIO(response.text)):
        cols = [str(c).strip() for c in tbl.columns]
        for cand in candidates:
            if cand in cols:
                return tbl[cand].dropna().astype(str).tolist()
    raise ValueError(f"Could not find a ticker column in the tables of {url}")


def _save_pct_returns(close: pd.DataFrame, out_path: str) -> None:
    pct_ret = close.pct_change().iloc[1:]
    pct_ret.dropna(axis="columns", inplace=True)
    os.makedirs(PATH_DATA, exist_ok=True)
    pct_ret.to_csv(out_path)
    print(f"Saved {len(pct_ret)} rows x {len(pct_ret.columns)} stocks to {out_path}")


def EuroStoxx50(start_date, end_date):
    firms_list = _wikipedia_tickers("https://en.wikipedia.org/wiki/EURO_STOXX_50",
                                    ["Ticker", "Symbol", "Ticker symbol", "Ticker(s)"])
    close = yf.download(firms_list, start=start_date, end=end_date)["Close"]
    _save_pct_returns(close, _out_path("euro50", start_date, end_date))


def FTSE100(start_date, end_date):
    firms_list = _wikipedia_tickers("https://en.wikipedia.org/wiki/FTSE_100_Index",
                                    ["Ticker", "Symbol", "EPIC", "Ticker symbol", "Ticker(s)"])
    close_first = yf.download(firms_list, start=start_date, end=end_date)["Close"]
    failed_tickers = close_first.columns[close_first.isna().any()].tolist()
    new_tickers = [s + ".L" for s in failed_tickers]
    if new_tickers:
        retry = yf.download(new_tickers, start=start_date, end=end_date)["Close"]
    else:
        retry = pd.DataFrame(index=close_first.index)
    close = close_first.dropna(axis="columns").join(retry, how="outer")
    _save_pct_returns(close, _out_path("ftse100", start_date, end_date))


def KOSPI200(start_date, end_date):
    # Expects data/raw/kospi200.csv (KRX constituent list) with a 6-digit code column named '종목코드'.
    data_path = os.path.join(PATH_RAW_DATA, 'kospi200.csv')
    raw_data = pd.read_csv(data_path, encoding="CP949")
    symbol_list = raw_data["종목코드"].tolist()
    dic = {}
    for symbol in tqdm(symbol_list):
        symbol = str(symbol).zfill(6)
        dic[symbol] = fdr.DataReader(symbol, start_date, end_date)['Change']
    pct_ret = pd.DataFrame(dic)
    pct_ret.dropna(axis='columns', inplace=True)
    os.makedirs(PATH_DATA, exist_ok=True)
    out_path = _out_path("kospi200", start_date, end_date)
    pct_ret.to_csv(out_path)
    print(f"Saved {len(pct_ret)} rows x {len(pct_ret.columns)} stocks to {out_path}")


_NIKKEI225_CODES = [
    1332, 1605, 1721, 1801, 1802, 1803, 1808, 1812, 1925, 1928, 1963,
    2002, 2269, 2282, 2413, 2432, 2501, 2502, 2503, 2531, 2768, 2801,
    2802, 2871, 2914, 3086, 3099, 3289, 3401, 3402, 3405, 3407, 3436,
    3659, 3861, 3863, 4004, 4005, 4021, 4042, 4043, 4061, 4063, 4151,
    4183, 4188, 4208, 4272, 4324, 4502, 4503, 4506, 4507, 4519, 4523,
    4543, 4568, 4578, 4661, 4689, 4704, 4755, 4901, 4902, 4911, 5002,
    5019, 5020, 5101, 5105, 5108, 5201, 5202, 5214, 5232, 5233, 5301,
    5332, 5333, 5401, 5406, 5411, 5541, 5631, 5703, 5706, 5707, 5713,
    5714, 5801, 5802, 5803, 5901, 6098, 6103, 6113, 6146, 6178, 6301,
    6302, 6305, 6326, 6361, 6367, 6376, 6412, 6471, 6472, 6473, 6501,
    6503, 6504, 6506, 6645, 6674, 6701, 6702, 6703, 6724, 6752, 6753,
    6758, 6762, 6770, 6841, 6857, 6902, 6952, 6954, 6971, 6976, 6981,
    7003, 7004, 7011, 7012, 7013, 7186, 7201, 7202, 7203, 7205, 7211,
    7261, 7267, 7269, 7270, 7272, 7731, 7733, 7735, 7751, 7752, 7762,
    7832, 7911, 7912, 7951, 8001, 8002, 8003, 8005, 8006, 8015, 8031,
    8035, 8053, 8058, 8233, 8252, 8253, 8267, 8306, 8308, 8309, 8316,
    8411, 8601, 8604, 8628, 8630, 8725, 8750, 8752, 8766, 8802, 8830,
    9001, 9005, 9007, 9008, 9009, 9021, 9022, 9064, 9101, 9104, 9107,
    9202, 9301, 9432, 9433, 9434, 9501, 9502, 9503, 9531, 9532, 9602,
    9613, 9681, 9735, 9766, 9983, 9984,
]


def Nikkei225(start_date, end_date):
    tickers = [str(c).zfill(4) + '.T' for c in _NIKKEI225_CODES]
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    close = df['Close'].copy()
    close.index.name = 'Date'
    close.dropna(axis='columns', how='all', inplace=True)
    pct_ret = close.pct_change().iloc[1:]
    pct_ret.dropna(axis='columns', inplace=True)
    pct_ret = pct_ret.loc[~(pct_ret == 0).all(axis=1)]
    os.makedirs(PATH_DATA, exist_ok=True)
    out_path = _out_path("nikkei225", start_date, end_date)
    pct_ret.to_csv(out_path)
    print(f"Saved {len(pct_ret)} rows x {len(pct_ret.columns)} stocks to {out_path}")


GENERATORS = {
    'euro50': EuroStoxx50,
    'ftse100': FTSE100,
    'kospi200': KOSPI200,
    'nikkei225': Nikkei225,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2016-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--data", type=str, choices=list(GENERATORS), default="euro50")
    args = parser.parse_args()
    print(f"\nGenerating {args.data.upper()} from {args.start} to {args.end} ...")
    GENERATORS[args.data](args.start, args.end)
