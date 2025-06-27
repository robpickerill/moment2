import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Parameters
TICKERS = pd.read_csv("./data/sp500_tickers.csv")["Symbol"].tolist()
TICKERS = [ticker.replace(".", "-") for ticker in TICKERS]
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=90)
LOOKBACK = 30  # Lookback window for lowest low to anchor AVWAP

trade_candidates = []

print(f"Fetching {len(TICKERS)} tickers from {START_DATE.date()} to {END_DATE.date()}")

for symbol in tqdm(TICKERS):
    data = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=True)
    if data is None or data.empty or "Close" not in data or "Volume" not in data:
        print(f"Skipping {symbol}: No data or missing columns")
        continue

    df = data.xs(symbol, level=1, axis=1).copy()
    df = df[["Close", "Low", "Volume"]].dropna()

    if len(df) < LOOKBACK + 5:
        print(f"Skipping {symbol}: Not enough data")
        continue

    anchor_idx = df["Low"].iloc[-LOOKBACK:].idxmin()

    anchored = df.loc[anchor_idx:].copy()
    anchored["TPV"] = anchored["Close"] * anchored["Volume"]
    anchored["AVWAP"] = anchored["TPV"].cumsum() / anchored["Volume"].cumsum()

    avwap_slope = anchored["AVWAP"].diff().iloc[-5:].mean()
    latest = anchored.iloc[-1]

    if (
        latest["Close"] > latest["AVWAP"]
        and avwap_slope > 0
        and latest["Close"] < latest["AVWAP"] * 1.05
    ):
        trade_candidates.append(
            {
                "Symbol": symbol,
                "Price": latest["Close"],
                "AVWAP": latest["AVWAP"],
                "AVWAP_Slope": avwap_slope,
                "Anchor_Date": pd.to_datetime(anchor_idx).strftime("%Y-%m-%d"),
            }
        )

if trade_candidates:
    trade_df = pd.DataFrame(trade_candidates).sort_values(
        by="AVWAP_Slope", ascending=False
    )
    print("\nTrade Candidates:")
    print(trade_df.to_string(index=False))
else:
    print("No trade candidates found.")
