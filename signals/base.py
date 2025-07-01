import pandas as pd

from abc import ABC, abstractmethod


class SignalBase(ABC):
    """
    Base class for signal generators.
    """

    @abstractmethod
    def generate(self, price: pd.Series, **kwargs) -> pd.Series:
        """
        Generate a signal based on the provided price series.
        """

        raise NotImplementedError("Must implement in subclass")
