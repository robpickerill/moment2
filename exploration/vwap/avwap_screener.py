import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.momentum import WilliamsRIndicator

# Parameters
TICKERS = pd.read_csv("./data/sp500_tickers.csv")["Symbol"].tolist()
TICKERS = [ticker.replace(".", "-") for ticker in TICKERS]
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=90)
LOOKBACK = 30  # Lookback window for lowest low to anchor AVWAP


def classify_row(row):
    price_diff_ratio = row["Price"] / row["AVWAP"]
    slope = row["AVWAP_Slope"]
    rsi = row["RSI"]
    macd_diff = row["MACD_Diff"]
    willr = row["WilliamsR"]

    if price_diff_ratio > 1.1:
        if slope > 1.5 and rsi > 70 and willr > -20:
            return "overextended"
        elif rsi > 75 or macd_diff < 0 or willr > -10:
            return "profit-taking zone"

    if slope > 2.5 and price_diff_ratio > 1.03:
        if macd_diff > 1 or willr > -10:
            return "event-driven"
        elif macd_diff < 0:
            return "event-fade"

    if 0.5 < slope <= 2.5 and 0.98 <= price_diff_ratio <= 1.05:
        if 50 <= rsi <= 70 and macd_diff > 0 and -80 < willr < -20:
            return "organic trend"
        elif 40 <= rsi < 50 and macd_diff > 0.2 and -90 < willr < -20:
            return "emerging trend"

    if slope < 0.4 or abs(macd_diff) < 0.1 or willr > -10:
        return "low momentum"

    return "low priority"


trade_candidates = []

print(f"Fetching {len(TICKERS)} tickers from {START_DATE.date()} to {END_DATE.date()}")

for symbol in tqdm(TICKERS):
    data = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=True)
    if data is None or data.empty or "Close" not in data or "Volume" not in data:
        print(f"Skipping {symbol}: No data or missing columns")
        continue

    df = data.xs(symbol, level=1, axis=1).copy()
    df = df[["Close", "High", "Low", "Volume"]].dropna()

    if len(df) < LOOKBACK + 5:
        print(f"Skipping {symbol}: Not enough data")
        continue

    df["rsi"] = RSIIndicator(close=df["Close"]).rsi()
    macd = MACD(close=df["Close"])
    df["macd_diff"] = macd.macd_diff()

    df["williams_r"] = WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"]
    ).williams_r()

    anchor_idx = df["Low"].iloc[-LOOKBACK:].idxmin()

    anchored = df.loc[anchor_idx:].copy()
    anchored["TPV"] = anchored["Close"] * anchored["Volume"]
    anchored["AVWAP"] = anchored["TPV"].cumsum() / anchored["Volume"].cumsum()

    avwap_slope = anchored["AVWAP"].diff().iloc[-5:].mean()
    latest = anchored.iloc[-1]

    """
      if:
        - price is rising above AVWAP
        - AVWAP slope is positive
        - price is within 5% of AVWAP
        - momentum is supported by MACD (MACD diff is positive)
        - RSI is not overbought (below 70)
        - Volume is positive
        - Williams %R demonstrates the stock is not overbought (above -80)
    """
    if (
        latest["Close"] > latest["AVWAP"]
        and avwap_slope > 0
        and latest["Close"] < latest["AVWAP"] * 1.05
        and latest["macd_diff"] > 0
        and latest["rsi"] < 70
        and latest["Volume"] > 0
        and latest["williams_r"] > -80
    ):
        trade_candidates.append(
            {
                "Symbol": symbol,
                "Price": latest["Close"],
                "AVWAP": latest["AVWAP"],
                "AVWAP_Slope": avwap_slope,
                "Anchor_Date": pd.to_datetime(anchor_idx).strftime("%Y-%m-%d"),
                "RSI": latest["rsi"],
                "MACD_Diff": latest["macd_diff"],
                "WilliamsR": latest["williams_r"],
                "Volume": latest["Volume"],
            }
        )


if trade_candidates:
    trade_df = pd.DataFrame(trade_candidates).sort_values(
        by="AVWAP_Slope", ascending=False
    )
    trade_df["Classification"] = trade_df.apply(classify_row, axis=1)

    print("\nTrade Candidates:")
    print(trade_df.to_string(index=False))
else:
    print("No trade candidates found.")
