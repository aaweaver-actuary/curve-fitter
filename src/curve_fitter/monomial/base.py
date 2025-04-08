from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import polars as pl

class BaseMonomial(ABC):
    @abstractmethod
    def __str__(self) -> str:
        """Return the monomial term formatted for the terminal."""
        pass

    @abstractmethod
    def to_html(self) -> str:
        """Return the monomial term formatted for html/markdown representation with MathJax. Defaults to str(self)."""
        return self.__str__()

    @abstractmethod
    def to_numpy(self, x: float) -> np.ndarray:
        """Return the base/numpy data type resulting from running x through the monomial term."""
        pass

    @abstractmethod
    def to_pandas(self, series: pd.Series) -> pd.Series:
        """Return the monomial term applied to every value in a Series. Overwrite this method if this should be different than the .to_numpy() method."""
        return series.apply(lambda x: self.to_numpy(x))

    @abstractmethod
    def to_polars(self, series: pd.Series) -> pd.Series:
        """Return the monomial term applied to every value in a Series. Overwrite this method if this should be different than the .to_numpy() method."""
        return series.apply(lambda x: self.to_numpy(x))