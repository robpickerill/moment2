import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from signals.base import SignalBase


class AVWAPR2Signal(SignalBase):
    def __init__(self, lookback: int = 30, r2_thresh: float = 0.8):
        self.lookback = lookback
        self.r2_thresh = r2_thresh

    def compute_avwap(self, price: pd.Series, volume: pd.Series) -> pd.Series:
        avwap = pd.Series(index=price.index, dtype=float)
        for t in price.index[self.lookback :]:
            window = price.loc[:t].iloc[-self.lookback :]
            anchor = window.idxmin()  # anchor to local low
            pr = price.loc[anchor:t]
            vol = volume.loc[anchor:t]
            tpv = (pr * vol).cumsum()
            cum_vol = vol.cumsum()
            avwap[t] = (
                tpv.iloc[-1] / cum_vol.iloc[-1] if cum_vol.iloc[-1] > 0 else np.nan
            )
        return avwap

    def compute_r2(self, series: pd.Series) -> pd.Series:
        r2_series = pd.Series(index=series.index, dtype=float)
        for t in series.index[self.lookback :]:
            y = series.loc[:t].iloc[-self.lookback :]
            if y.isna().any():
                continue
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y.values)
            r2_series[t] = model.score(x, y.values)
        return r2_series

    def generate(self, price: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
        avwap = self.compute_avwap(price, volume)
        r2 = self.compute_r2(avwap)
        signal = r2 > self.r2_thresh
        return signal.astype(bool)
