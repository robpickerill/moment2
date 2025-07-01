import pandas as pd


class AVWAPCalculator:
    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def compute(self, price: pd.Series, volume: pd.Series) -> pd.Series:
        avwap = pd.Series(index=price.index, dtype=float)
        for t in price.index[self.lookback :]:
            window = price.loc[:t].iloc[-self.lookback :]
            anchor = window.idxmin()  # or idxmax()
            pr = price.loc[anchor:t]
            vol = volume.loc[anchor:t]
            tpv = (pr * vol).cumsum()
            cum_vol = vol.cumsum()
            avwap[t] = (
                tpv.iloc[-1] / cum_vol.iloc[-1] if cum_vol.iloc[-1] > 0 else np.nan
            )
        return avwap
