import pandas as pd
from datetime import datetime,timedelta,time

def difference(data,h):
    """
    difference - given a pandas series data, return the 
                values shifted by h hours used by other 
                methods (conga-h)
    Input: data - pandas Series with index as a datetime, 
                values are glucose readings associated 
                with those times.
    Output: pandas Series differenced and shifted by h hours
    """
    index = data.index
    idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=h*12)
    idx1 = idx_shift[idx_shift.isin(index)]
    idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=-h*12)
    idx2 = idx_shift[idx_shift.isin(index)]
    diff = []
    for i in range(len(idx2)):
        diff.append(data[idx1[i]]-data[idx2[i]])
    return pd.Series(diff,index=idx1,dtype=float)

def difference_m(data,m):
    """
    difference_m - given a pandas series data, return the 
                difference shifted by m minutes used by 
                variability metrics.
    Input: data - pandas Series with index as a datetime, 
                values are glucose readings associated with 
                those times.
    Output: pandas Series diffenced and shifted by m minutes
    """
    index = data.index
    period = m//5
    idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=period)
    idx1 = idx_shift[idx_shift.isin(index)]
    idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=-period)
    idx2 = idx_shift[idx_shift.isin(index)]
    diff = []
    for i in range(len(idx2)):
        diff.append(data[idx1[i]]-data[idx2[i]])
    return pd.Series(diff,index=idx1,dtype=float)  