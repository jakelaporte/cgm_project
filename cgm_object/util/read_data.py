import pandas as pd
import numpy as np
import os
from datetime import datetime,timedelta,time
import streamlit as st
from .helper import tic, itc

def generate_range(start_date, end_date):
    current = [start_date]
    delta = timedelta(minutes=5)
    while current[-1] != end_date:
        current.append(current[-1]+delta)
    return current


def get_folder(curr_dir):
    pass



def build_periods(series):
    ts = list(series.index)
    series['dates']=list(series.index)
    series['date_shift']=series['dates'].shift(-1)
    series['time_diff']=(series['date_shift']-series['dates'])/pd.Timedelta(minutes=1)
    idxs = list(series[series['time_diff']>1440][['dates','date_shift']].values)
    periods = [[ts[0],ts[-1]]]
    for idx in idxs:
        t0 = pd.Timestamp(idx[0])
        t1 = pd.Timestamp(idx[1])
        periods.append([t1,periods[-1][1]])
        periods[-2][1]=t0
    periods_ = []
    for per in periods:
        t0 = series.loc[per[0]:per[1]]['glucose'].first_valid_index()
        t1 = series.loc[per[0]:per[1]]['glucose'].last_valid_index()
        periods_.append([t0,t1])
    return periods_

def return_data(df,col,day):
    """
    return_data - given a dataframe df with index of datetimes, a column (col)
        of interest and a particular day (day) -> return a series containing the 
        datetime index and values from the column associated with that day.
        
    Input:  df - dataframe with index as datetime
            col - a column in the given dataframe
            day - (string) a day in the index of the dataframe or list of days
    
    Output: series with an index of datetimes and values of the column of interest.
    """
    df['day']=list(df.loc[:].index)
    days=df['day'].apply(tic)
    df['day'] = [d[0] for d in days.values]
    try:
        vals = df.loc[df['day']==day][col]
    except:
        vals = df.loc[df['day'].isin(day)][col]
    return vals

def return_time_data(series,time0,time1):
    """
    return_time_data - returns imputed glucose values for all days 
        between time0 and time1.
    """
    fmt = '%m/%d/%Y-'+'%H:%M'
    df = series.copy()
    day0 = series.index[0].date().strftime("%m/%d/%Y")
    time = [t.time() for t in df.index]
    df['time']=time
    time0 = datetime.strptime(day0+'-'+time0,fmt).time()
    time1 = datetime.strptime(day0+'-'+time1,fmt).time()
    df = df[(df['time']>=time0) & (df['time']<=time1)]

    return df['imputed']

def all_day_data(day,**kwargs):
    """
    returns all imputed data from start_time to end_time for a day.
    
    Input: day - the day in string forma
    """

    series = kwargs['series'].copy()
    dt = itc(day,0)
    vals = series.loc[series['day']==dt.date()]['imputed']
    return vals
    
def all_data(series,impute_):
    """
    all_data - returns a series of all of the data based on impute or not to impute.
    
    Input: impute_ -> True (imputed glucose values) or False (original glucose values)
    
    Output: pandas series with index as datetime and values as glucose values.
    """
    if impute_:
        return series.loc[:,'imputed']
    else:
        return series.loc[:,'glucose']
    
def return_period_data(series,periods,impute_):
    """
    returns all of the data from each period strung together so datetimes will not be continuous 
        between periods.
    """
    if impute_:
        rtn_data = pd.DataFrame()
        for period in periods:
            period_data = series.loc[period[0]:period[1],'imputed']
            if len(rtn_data) == 0:
                rtn_data = pd.DataFrame(period_data.copy())
            else:
                rtn_data = pd.concat([rtn_data,period_data])
    else:
        rtn_data = pd.DataFrame()
        for period in periods:
            period_data = series.loc[period[0]:period[1],'glucose']
            if len(rtn_data) == 0:
                rtn_data = pd.DataFrame(period_data.copy())
            else:
                rtn_data = pd.concat([rtn_data,period_data])
    #st.write(rtn_data)
    
    return pd.Series(rtn_data.iloc[:,0].values,index = rtn_data.index)


def read_data(filename,dt_col=1,gl_col=7,dt_fmt='%Y-%m-%dT%H:%M:%S',skip_rows=1):
    def adjust_times(x):
        dt,tm =x.split(' ') 
        tm = tm.split(':')
        tm[0]=int(tm[0])
        tm[1]=int(tm[1])//5*5
        tm[2]=0
        tm = time(tm[0],tm[1])
        dt = datetime.strptime(dt,'%Y-%m-%d')
        return datetime.combine(dt,tm)
    try:
        """
        output is a dataframe with date time index and glucose as values in a column
        """
        data = pd.read_csv(filename)
        data = data.iloc[:,[dt_col,gl_col]]
        data = data.dropna()
        data['index']=pd.to_datetime(data.iloc[:,0],format=dt_fmt)
        data['index']=data['index'].astype(str)
        data['index'] = data['index'].map(adjust_times)
        data = data.set_index('index')
        cols = data.columns    
        data.index.name = 'datetime'
        data = data[[cols[1]]]
        data.columns = ['glucose']
    except:
        data = read_io_data(filename,dt_col=dt_col,gl_col=gl_col,
                            dt_fmt=dt_fmt,skip_rows=skip_rows)
    st.write(data)
    periods = build_periods(data.copy())
    st.write(periods)
    series = pd.DataFrame()
    #get rid of duplicated datetimes
    data = data[~data.index.duplicated()]
    for period in periods:
        temp = pd.DataFrame([],index = generate_range(period[0],period[1]))
        series = pd.concat([series,temp])
    
    series['glucose']=data['glucose']
    series['day'] = series.index.map(lambda t: t.date())
    series['time'] = series.index.map(lambda t: t.time())
    series['min'] = series['time'].astype(str).str.split(":").map(lambda t: int(t[0])*60+int(t[1]))
    return series,periods


def read_io_data(filename,dt_col=1,gl_col=7,dt_fmt='%Y-%m-%dT%H:%M:%S',skip_rows=1):
    """
    filename is a string IO from streamlit, import the data for the 
        glucose readings and the datetime for those readings into a dataframe
        that can be used by read_data.
        
    Input: filename - is actually a stream of IO data from the file 'filename'
           dt_fmt - the datetime format for the datetime data.
           dt_col - the column that the datetime appear.
           gl_col - the column that the glucose value appears.
    """
    def round_down_date_time(x):
        minute = x.minute//5*5
        x = x.replace(minute = minute)
        x = x.replace(second = 0,microsecond=0)
        return x
    from io import StringIO
    start_date = '2000-01-01'
    infile = StringIO(filename.decode("utf-8"))
    #infile = open(filename)
    for i in range(skip_rows):
        _ = infile.readline()
    data={}
    dates = []
    for line in infile:
        row = line.split(',')
        dt = row[dt_col]
        dt = dt.replace('"','')
        try:
            dt = datetime.strptime(dt,dt_fmt)
            dt = round_down_date_time(dt)
            val = row[gl_col]
            data[dt]=int(float(val.rstrip()))
        except:
            pass
        dates.append(dt)
        val = row[gl_col]
        try:
            data[dt]=int(float(val.rstrip()))
        except:
            data[dt]=np.nan

    infile.close()
    data = pd.Series(data)
    data = pd.DataFrame(data,columns = ['glucose'])
    return data


