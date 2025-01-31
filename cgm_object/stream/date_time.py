from datetime import datetime,timedelta
import streamlit as st
from io import StringIO
import numpy as np
import pandas as pd
from ..util import generate_range

def view_data(filename,dt_col,gl_col,dt_fmt):
    st.write(dt_fmt)
    infile = StringIO(filename.decode("utf-8"))
    header = infile.readline()
    data = {}
    for i,line in enumerate(infile):
        row = line.split(',')
        dt = row[dt_col]
        dt=dt.replace('"','')
        #st.write(dt)
        dt = datetime.strptime(dt,dt_fmt)
        #st.write(dt)
        minute = dt.minute
        dt=dt.replace(minute=minute//5*5,second=0)
        try:
            val = row[gl_col]
            val = int(float(val.rstrip()))
        except:
            val = np.nan
        while dt in data.keys():
            day = dt.day
            day += 1
            try:
                dt = dt.replace(day=day)
            except:
                pass
        if i > 1000:
            break
        data[dt]=val
    data = pd.DataFrame(pd.Series(data))
    data.columns=['glucose']
    data.index.name = 'datetime'
    st.write(data)

def transform_data(filename,dt_col,gl_col,dt_fmt,start_date):
    infile = StringIO(filename.decode("utf-8"))
    start_date = datetime.strptime(start_date.strftime("%Y-%m-%d"),"%Y-%m-%d").date()
    header = infile.readline()
    data = {}
    
    for i,line in enumerate(infile):
        row = line.split(',')
        dt = row[dt_col]
        dt = dt.replace('"','')
        try:
            dt = datetime.strptime(dt,'%H:%M:%S').time()
        except:
            dt = datetime.strptime(dt,dt_fmt).time()
        dt = datetime.combine(start_date,dt)
        minute = dt.minute
        dt=dt.replace(minute=minute//5*5,second=0)
        try:
            val = row[gl_col]
            val = int(float(val.rstrip()))
        except:
            val = np.nan
        while dt in data.keys():
            dt += timedelta(days=1)
        
        data[dt]=val
        if i > 1000:
            break
    data = pd.DataFrame(pd.Series(data))
    data.columns=['glucose']
    data.index.name = 'datetime'
    return data

def view_raw_data(filename):
    infile = StringIO(filename.decode("utf-8"))
    header = infile.readline().split(',')
    data={}
    for i in range(len(header)):
        data[header[i]]=[]
    for j,line in enumerate(infile):
        row = line.split(',')
        for i,col in enumerate(row):
            data[header[i]].append(col)
        if j == 4:
            break
    return pd.DataFrame(data)






