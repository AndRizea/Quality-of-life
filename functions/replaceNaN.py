import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

def replaceNAN(t):
    assert isinstance(t, pd.DataFrame)
    vars = list(t.columns)
    for var in vars:
        if any(t[var].isna()):
            if is_numeric_dtype(t[var]):
                t[var].fillna(t[var].mean(), inplace=True)
                t[var] = np.round(t[var], 1)
            else:
                mode = t[var].mode()[0]
                t[var].fillna(mode, inplace=True)
                t[var] = np.round(t[var], 1)