# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 07:10:59 2025

@author: grover.laporte
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from scipy.stats import t
from scipy.signal import argrelextrema
from datetime import datetime,timedelta
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
#from .util import *
from .util import (read_data,return_period_data,
                   return_time_data,unique_days,
                   linear_interpolate,return_data,
                   )
#from .functions import * 
from .functions import (time_in_range,glucose_N,glucose_mean,
                        glucose_std,glucose_cv,mean_absolute_glucose,
                        j_index,low_high_blood_glucose_index,
                        glycemic_risk_assessment_diabetes_equation,
                        glycemic_variability_percentage,
                        lability_index,mean_of_daily_differences,
                        average_daily_risk_range,conga,m_value,
                        glucose_management_indicator,interquartile_range,
                        auc,mean_amplitude_of_glycemic_excursions,
                        glycemic_risk_index,percent_active,auc_5,
                        adrr_easy,)

class CGM(object):
    def __init__(self,filename,dt_col=1,gl_col=7):
        """
        Dexcom data object for visualizing, imputing, and using dexcom 
            continuous glucose monitoring data.
            
        filename - input of bytes of data ... still figuring out the best 
                    method.
            
        data - dictionary of values with keys as days in the file; and values
            as a list of lists for each record which has a list of
            2 components: 1) # of minutes after midnight according to time
            stamp and 2) glucose reading
        df - dataframe with the same information as data just formatted into
            a table 0:1440:5 as the columns which represent the times, the
            index as dates with the format 'mm/dd/yyy' and the values in the
            table as glucose values.
            
        start_time - first datetime with a valid glucose value.
        
        end_time - last datetime with a valid glucose value.
            
        series - dataframe with one column of all the glucose values with 
            a datetime as the index and 'glucose' values as the one column
            
        days - the days in the format 'mm/dd/yyyy' from the time stamp
        
        times - the 288 times which represent the number of minutes 
            after midnight [0,5,10,...1440]
            
        shape - the number of days (rows) and times (columns) in the data
        
        rows - the days in the data that have missing data.
            Example: [0,3,8] day 0,3,8 are missing at least one instance
                of data
        cols - a list of columns from the data that is missing data. These
            will align with rows to give the columns in each row that are 
            missing.
            Example: [array([0,1,2,...238]), array(230,231), 
                      array(159,160,...287)] day 0 is missing the first part
                    of the day, day 3 is missing only the 230 and 231 columns.
        
        """
        self.filename = filename
        self.series,self.periods = read_data(filename,dt_col=dt_col,gl_col=gl_col)
        self.times = np.array(list(np.arange(0,1440,5)))
        self.build_daily_dataframe()
        
        X = self.df.copy()
        rows = X.isnull().any(axis=1)
        self.impute_bool = True if rows.sum()>0 else False
        rows = np.where(rows==True)[0]
        self.rows = rows
        cols = []
        for row in rows:
            col_ = np.where(X.iloc[row].isnull()==True)[0]
            cols.append(col_)
        self.cols = cols
        self.missing_data={}
        for i,r in enumerate(self.rows):
            day = self.days[r]
            self.missing_data[day]={}
            self.missing_data[day]['row']=r
            self.missing_data[day]['cols']=self.cols[i]
            self.missing_data[day]['impute']= True 
        self.stats = {}; self.params={}
        
        data = return_period_data(self.series,self.periods,True)
        
        self.set_params({'type':'paper','unit':'mg','lower':70,'conga_h':1,
                         'highest':250,'upper':180,'lowest':54,'thresh':100,
                         'above':True,'m_index':120,'li_k':60,'N':len(data),
                         'nans':self.nans,
                         'days':self.days,
                         'series':self.series.copy(),
                         'data':data,
                         'day_data':return_time_data(self.series.copy(),'6:00','23:55'),
                         'night_data':return_time_data(self.series.copy(),'00:00','06:00')})
        self.N = len(data)
        self.assign_functions()
        self.calculate_stats()
        
    ### Pulling in the data from the file ####################################
    ### Creating attributes that can explain features of the data ############
    ### data - dictionary, df - dataframe, series - series ###################
    
    
    def build_daily_dataframe(self):
        series = self.series.copy()
        self.df = series.pivot_table(values='glucose',index='day',columns='min')
        self.days = unique_days(series)
        total_time = timedelta(0)
        total_seconds=total_minutes=total_hours=total_days=nans=0
        for period in self.periods:
            ser = series.loc[period[0]:period[1],['glucose']]
            time = period[1]-period[0]
            total_time += time
            seconds = time.total_seconds()
            total_seconds += seconds
            minutes = seconds/60
            total_minutes += minutes
            hours = minutes/60
            total_hours += hours
            days = hours/24
            total_days += days
            indices = np.where(ser.isnull())[0]
            nans+=len(indices)
            series.loc[period[0]:period[1],'imputed'] = linear_interpolate(ser['glucose'].values,indices)
        
        self.series=series.copy()
        self.start_time = self.periods[0][0]
        self.end_time = self.periods[-1][1]
        self.nans = nans
        self.total_time = total_time
        self.total_seconds = total_seconds
        self.total_minutes = total_minutes
        self.total_hours = total_hours
        self.total_days = total_days
        
        return None
    
    ###########################################################################
    ### Helper functions for time conversions, linear interpolation, rtc ######
    ###########################################################################
    
    # def convert_to_time(self,minutes):
    #     """
    #     given a list of minutes after midnight, convert to a string time.
    #     """
    #     time = []
    #     for m in minutes:
    #         hh = m//60
    #         mm = m%60
    #         time.append(f'{hh:0>2}:{mm:0>2}')
    #     return time
    
    # def generate_range(self,start_date, end_date):
    #     current = [start_date]
    #     delta = timedelta(minutes=5)
    #     while current[-1] != end_date:
    #         current.append(current[-1]+delta)
    #     return current
    
    # def linear_interpolate(self, glucose_values, indices):
    #     """
    #     linear_interpolate - interpolating values between data
        
    #     Input:
    #         glucose_values - all of the glucose values for a day (288)
    #         indices - the indicies of the glucose values that need imputing
    #     Output:
    #         interpolated_values - all of the glucose values linear interpolated
    #         for those that are missing.
    #     """
    #     x = np.array(np.arange(len(glucose_values)),dtype=int)
    #     mask = np.ones(len(glucose_values), dtype=bool)
    #     mask[indices] = False
    #     interpolated_values = np.interp(x, 
    #                                     x[mask], 
    #                                     glucose_values[mask].astype(int))
    #     return interpolated_values
    
    
    # def itc(self,day,mm):
    #     """
    #     itc - (index_time_conversion) - converts a day / time (min past midnight)  
    #         to a datetime needed to index the series
    #     """
    #     return datetime.strptime(day+f' {mm//60:0>2}:{mm%60:0>2}','%m/%d/%Y %H:%M')
    
    # def tic(self,idx):
    #     """
    #     tic - (time_index_conversion) - converts a datetime to day and min past
    #         midnight in a day to access self.df
    #     """
    #     day = idx.strftime('%m/%d/%Y')
    #     time = idx.strftime('%H:%M')
    #     return day,time,int(time[:2])*60+int(time[3:])
    
    # def unique_days(self,data):
    #     """
    #     unique_days - given a pandas Series with datetime as it's index => return
    #                   the unique days in that set.
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
                      
    #     Output: days - a list of unique days.
    #     """
    #     days = [self.tic(d)[0] for d in data.index]
    #     days = sorted(list(set(days)))
    #     return days
    
    ########################################################################################
    ### Helper functions that return data ##################################################
    ########################################################################################
    
    # def return_data(self,df,col,day):
    #     """
    #     return_data - given a dataframe df with index of datetimes, a column (col)
    #         of interest and a particular day (day) -> return a series containing the 
    #         datetime index and values from the column associated with that day.
            
    #     Input:  df - dataframe with index as datetime
    #             col - a column in the given dataframe
    #             day - (string) a day in the index of the dataframe or list of days
        
    #     Output: series with an index of datetimes and values of the column of interest.
    #     """
    #     df['day']=list(df.index)
    #     days=df['day'].apply(tic)
    #     df['day'] = [d[0] for d in days.values]
    #     try:
    #         vals = df.loc[df['day']==day][col]
    #     except:
    #         vals = df.loc[df['day'].isin(day)][col]
    #     return vals
    
    # def return_time_data(self,time0,time1):
    #     """
    #     return_time_data - returns imputed glucose values for all days 
    #         between time0 and time1.
    #     """
    #     fmt = '%m/%d/%Y-'+'%H:%M'
    #     df = self.series.loc[self.start_time:self.end_time].copy()
    #     time = [t.time() for t in df.index]
    #     df['time']=time
    #     time0 = datetime.strptime(self.days[0]+'-'+time0,fmt).time()
    #     time1 = datetime.strptime(self.days[0]+'-'+time1,fmt).time()
    #     df = df[(df['time']>=time0) & (df['time']<=time1)]
    
    #     return df['imputed']
        
    
    # def change_data(self,day,impute_):
    #     """
    #     change_data - for a particular day, gives the original or imputed data based on 
    #     the value for impute_
        
    #     Input: day -> "10/16/2024" for example as a string
    #            impute_ -> True (imputed data) or False (original data)
               
    #     Output: returns the glucose values (or imputed glucose values) for the day.
    #     """
    #     dt = self.itc(day,0)
        
    #     if impute_:
    #         vals = self.series.loc[self.series['day']==dt.date()]['imputed'].values    
    #         #vals = np.rint(vals).astype(int)
    #     else:
    #         vals = self.series.loc[self.series['day']==dt.date()]['glucose'].values
    #     ## changes the self.df dataframe based on impute_ = True or False
    #     self.df.loc[day,self.times]=vals
    #     return vals
    
    # def all_day_data(self,day):
    #     """
    #     returns all imputed data from start_time to end_time for a day.
        
    #     Input: day - the day in string forma
    #     """
    #     dt = self.itc(day,0)
    #     vals = self.series.loc[self.series['day']==dt.date()]['imputed']
    #     return vals.loc[self.start_time:self.end_time]
        
    # def all_data(self,impute_):
    #     """
    #     all_data - returns a series of all of the data based on impute or not to impute.
        
    #     Input: impute_ -> True (imputed glucose values) or False (original glucose values)
        
    #     Output: pandas series with index as datetime and values as glucose values.
    #     """
    #     if impute_:
    #         return self.series.loc[self.start_time:self.end_time,'imputed']
    #     else:
    #         return self.series.loc[self.start_time:self.end_time,'glucose']
        
    # def return_period_data(self,impute_):
    #     """
    #     returns all of the data from each period strung together so datetimes will not be continuous 
    #         between periods.
    #     """
    #     if impute_:
    #         rtn_data = pd.DataFrame()
    #         for period in self.periods:
    #             period_data = self.series.loc[period[0]:period[1],'imputed']
    #             rtn_data = pd.concat([rtn_data,period_data])
    #     else:
    #         rtn_data = pd.DataFrame()
    #         for period in self.periods:
    #             period_data = self.series.loc[period[0]:period[1],'glucose']
    #             rtn_data = pd.concat([rtn_data,period_data])
                
    #     return pd.Series(rtn_data.iloc[:,0].values,index = rtn_data.index)
    
    def missing_criteria(self,day):
        """
        Once we determine the criteria for imputing automatically, we would 
            write that in here.Currently, we are only checking for max number
            of missing records and if the day is at the beginning or end.
            
        Work still needs to be done here.
        """
        MAX_MISSING = 30
        criteria_met = False
        ## get the days total_missing and the missing_day in context
        ## with the other days in the data (which day - first or last)
        total_missing = len(self.missing_data[day]['cols'])
        missing_day = self.missing_data[day]['row']
        
        ## Determine the total number of missing elements
        if total_missing>0 and total_missing<MAX_MISSING:
            criteria_met = True
            
        ## We do not want to extrapolate outside the wear window
        if missing_day==0 or missing_day == len(self.days)-1:
            criteria_met = False
            
        return criteria_met
    
    #######################################################################
    ### Differencing Functions ############################################
    #######################################################################
    # def difference(self,data,h):
    #     """
    #     difference - given a pandas series data, return the 
    #                 values shifted by h hours used by other 
    #                 methods (conga-h)
    #     Input: data - pandas Series with index as a datetime, 
    #                 values are glucose readings associated 
    #                 with those times.
    #     Output: pandas Series differenced and shifted by h hours
    #     """
    #     index = data.index
    #     idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=h*12)
    #     idx1 = idx_shift[idx_shift.isin(index)]
    #     idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=-h*12)
    #     idx2 = idx_shift[idx_shift.isin(index)]
    #     diff = []
    #     for i in range(len(idx2)):
    #         diff.append(data[idx1[i]]-data[idx2[i]])
    #     return pd.Series(diff,index=idx1,dtype=float)
    
    # def difference_m(self,data,m):
    #     """
    #     difference_m - given a pandas series data, return the 
    #                 difference shifted by m minutes used by 
    #                 variability metrics.
    #     Input: data - pandas Series with index as a datetime, 
    #                 values are glucose readings associated with 
    #                 those times.
    #     Output: pandas Series diffenced and shifted by m minutes
    #     """
    #     index = data.index
    #     period = m//5
    #     idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=period)
    #     idx1 = idx_shift[idx_shift.isin(index)]
    #     idx_shift = data.index.shift(freq=timedelta(minutes=5),periods=-period)
    #     idx2 = idx_shift[idx_shift.isin(index)]
    #     diff = []
    #     for i in range(len(idx2)):
    #         diff.append(data[idx1[i]]-data[idx2[i]])
    #     return pd.Series(diff,index=idx1,dtype=float)    
        
                
    #######################################################################
    ### Variability Metrics ###############################################
    #######################################################################

    # def time_in_range(self,**kwargs):
    #     """
    #     time in range - assumes 5 minute intervals for all data and simply
    #         counts the number between lower and upper / total number of 
    #         observations.
    #     Input:  data - needs to be a series object with either datetime or 
    #         minutes after midnight as the index.
    #             lower - the lower bound to check
    #             upper - the upper bound to check
    #     Output: a tuple of floats that represent the 
    #         (%time below range, %time in range, %time above range)
            
    #     """
    #     data = kwargs['data']
    #     lowest = kwargs['lowest']
    #     lower = kwargs['lower']
    #     upper = kwargs['upper']
    #     highest = kwargs['highest']
    #     data = data[data.notnull().values]
    #     denom = len(data)
        
    #     below_54 = len(data[data<lowest])
    #     below_range = len(data[data<lower])-below_54
    #     in_range = len(data[(data>=lower) & (data<=upper)])
    #     above_250 = len(data[data>highest])
    #     above_range = len(data[data>upper])-above_250
        
    #     res = np.array([below_54,below_range,in_range,above_range,above_250])
    #     return res/denom
    
    # def find_min(self,data):
    #     data = data[~data.isnull().values]
    #     idx = argrelextrema(data.values, np.less,order=5)[0]
    #     return data[idx]
    
    # def find_max(self,data):
    #     data = data[~data.isnull().values]
    #     idx = argrelextrema(data.values, np.greater,order=5)[0]
    #     return data[idx]
    
    # def conga(self,**kwargs):
    #     """
    #     conga - continuous overall net glycemic action (CONGA) McDonnell paper 
    #             and updated by Olawsky paper
                
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #            h - number of hours to shift
    #            type_ - "paper" or "easy" easy is the update.
               
    #     Output: CONGA(h) as a float.
    #     """
    #     data = kwargs['data']
    #     type_ = kwargs['type']
    #     h = kwargs['conga_h']
    #     sample_rate = 5 #in minutes      
    #     samples_per_hour = 60//sample_rate
    #     ## shift method moves values back and forward
    #     ## line1 moves the values an hour ahead back so 
    #     ## they can be used below
    #     line1 = data.shift(-samples_per_hour*h).dropna()
    #     delta = self.difference(data.dropna(),h)
    #     if type_ == 'paper':
    #         congah = delta.std()
    #         return congah
    #     if type_ == 'easy':
    #         k = len(delta)
    #         d_star = (abs(delta)).sum()/k
    #         congah = np.sqrt(((line1-d_star)**2).sum()/(k-1))
    #         return congah
    #     return None
    
    # def lability_index(self,**kwargs):
    #     """
    #     lability_index - for glucose measurement at time Xt, Dt = difference
    #         of glucose measurement k minutes prior.
    #     Input:  data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #             k - length of time in minutes (5 min increments) used to find patterns
    #     Output: LI as a float.
    #     """
    #     data = kwargs['data']
    #     k = kwargs['li_k']
    #     Dt = self.difference_m(data,k)
    #     try: #if there are too few data values for the data given
    #         li = (Dt**2).sum()/(len(Dt))
    #     except:
    #         li = np.nan
    #     return li
        
    # def mean_absolute_glucose(self,**kwargs):
    #     """
    #     mean_absolute_glucose - Hermanides (2009) paper
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #     Output: MAG as a float.
    #     """
    #     data = kwargs['data']
    #     total_hours = (data.index[-1]-data.index[0]).total_seconds()/3600
    #     data = data[~data.isnull().values].values
    #     diff = np.abs(data[1:]-data[:-1])
    #     return diff.sum()/(total_hours)
    
    # def glycemic_variability_percentage(self,**kwargs):
    #     """
    #     glycemic_variability_percentage - Peyser paper length of curve / length
    #                     straight line with no movement (time[final]-time[initial])
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #     Output: GVP as a float percentage.
    #     """
    #     data = kwargs['data']
    #     data = data[~data.isnull().values]
    #     time = data.index
    #     data = data.values
    #     t2 = [((time[i+1]-time[i]).total_seconds()/60)**2 for i in range(len(time)-1)] 
    #     y2 = (data[1:]-data[:-1])**2
    #     seg = np.array(t2+y2)
    #     L = np.array([np.sqrt(a) for a in seg]).sum()
    #     L0 = np.sqrt(t2).sum()
    #     return (L/L0-1)*100
    
    # def j_index(self,**kwargs):
    #     """
    #     j_index - calculates J-index 

    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #     Output: J-index as a float.
    #     """
    #     data = kwargs['data']
    #     unit = kwargs['unit']
    #     if unit == 'mg':
    #         return (data.mean()+data.std())**2/1000
    #     if unit =="mmol":
    #         return (18**2)*(data.mean()+data.std())**2/1000
    #     return None
    
    # def low_high_blood_glucose_index(self,**kwargs):
    #     """
    #     low_high_blood_glucose_index - calculates the blood glucose index 
    #                 with three sets of indices.
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #            type_- "paper", "easy", or "update" Default = "paper"
    #            unit - "mg" or "mmol" Default: "mg"
    #     """
    #     data = kwargs['data']
    #     type_ = kwargs['type']
    #     unit = kwargs['unit']
    #     n = len(data)
    #     data = data[~data.isnull().values].values
    #     f = 1
    #     c = 1
    #     if unit == 'mg':
    #         f = 1.509*(np.log(data)**1.084-5.381)
    #     if unit == 'mmol':
    #         f = 1.509*(np.log(18*data)**1.084-5.381)
    #     if type_ == 'update':
    #         c = 22.77
    #     if type_ == 'paper':
    #         c = 10
    #     if type_ == 'easy':
    #         c = 10
    #     rl = np.array([c*r**2 if r<0 else 0 for r in f])
    #     rh = np.array([c*r**2 if r>0 else 0 for r in f])
    #     if type_ != 'easy':
    #         nl = n
    #         nh = n
    #     else:
    #         nl=(rl>0).sum()
    #         nh=(rh>0).sum()
    #     return rl.sum()/nl, rh.sum()/nh
    
    # def glycemic_risk_assessment_diabetes_equation(self,**kwargs):
    #     """
    #     GRADE - or glycemic risk assessment diabetes equation

    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #            type_ - "paper" or "easy" Default: "paper"
    #            unit - "mg" or "mmol" Default: "mg"

    #     Output: GRADE as 4 numbers ================================= 
    #             (1) GRADE or mean/median of conversion, 
    #             (2) GRADE for values < 70.2(mg) or 3.9(mmol), 
    #             (3) GRADE for values between 70.2(mg) or 3.9(mmol) 
    #                                     and 140.4(mg) or 7.8(mmol),
    #             (4) GRADE for values above 140.4(mg) or 7.8(mmol)
    #     """
    #     data = kwargs['data']
    #     type_ = kwargs['type']
    #     unit = kwargs['unit']
    #     g = data[~data.isnull().values].values
    #     c1,c2 = 3.9,7.8
    #     if unit == 'mg':
    #         g = g/18
    #     if type_=='paper':
    #         c = 0.16
    #     if type_ == 'easy':
    #         c = 0.15554147
    #     h_log = lambda x,c: 425*((np.log10(np.log10(x))+c)**2)
    #     h_min = lambda x: x*(x<50)+50*(x>=50)
    #     h = lambda x,c: h_min(h_log(x,c))
    #     h_i = h(g,c)

    #     # separate glucose values into categories based on value
    #     gl = g[g<c1]
    #     gm = g[(c1<g)&(g<c2)]
    #     gh = g[g>c2]

    #     # run each group of glucose values through the functions
    #     hl = h(gl,c)
    #     hm = h(gm,c)
    #     hh = h(gh,c)
    #     h_sum = h_i.sum()
    #     if type_ == 'easy':
    #         grade = np.median(h_i)
    #     if type_ == 'paper':
    #         grade = h_i.mean()
    #     ans = np.array([grade,hl.sum()/h_sum,hm.sum()/h_sum,hh.sum()/h_sum])
    #     return ans    
    
    # def mean_amplitude_of_glycemic_excursions(self,**kwargs):
    #     """
    #     MAGE (Olawsky 2019)
    #     mean_amplitude_of_glycemic_excursions - MAGE mean of differences that are
    #         large compared to daily value.
    #     """
    #     data = kwargs['data']
    #     data = data[~data.isnull().values]
    #     E = []
    #     for day in self.days:
    #         # g - glucose values for day=day
    #         g = self.all_day_data(day,**kwargs)
    #         # s - standard deviation for glucose on day=day
    #         s = g.std()
    #         # D - glucose values differenced (5 minutes)
    #         D = self.difference_m(g,5)
    #         # test if abs(d) > standard deviation for the day
    #         for d in D:
    #             if abs(d)>s:
    #                 E.append(d)
    #     ## Use numpy array to sort / find mean of data
    #     if len(E)>0:
    #         E = np.array(E)
    #         mage_plus = E[E>0].mean()
    #         mage_minus = E[E<0].mean()
    #     else:
    #         mage_plus = mage_minus = np.nan
    #     return mage_plus,mage_minus
    
    # def mean_of_daily_differences(self,**kwargs):
    #     """
    #     MODD - or mean of daily differences
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #            type_ - algorithm to use - either "paper" or "easy" 
    #     Output: MODD as a float
    #     """
    #     data = kwargs['data']
    #     type_ = kwargs['type']
    #     days = kwargs['days']
    #     data = data[~data.isnull().values]
    #     #days = unique_days(data)
    #     if len(days)>=2:
    #         delta = self.difference(data,24)
    #         if type_ == 'paper':
    #             return (abs(delta)).sum()/len(delta)
    #         if type_ == 'easy':
    #             delta = delta[delta != delta.max()]
    #             return (abs(delta)).sum()/(len(delta))
    #     else:
    #         return np.nan
        
    # def average_daily_risk_range(self,**kwargs):
    #     """
    #     average_daily_risk_range - returns ADRR based on actual days. See below.
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #     Output: ADRR as three values - sum of low and high, low risk rate, high risk rate.
    #     """
    #     data = kwargs['data']
    #     unit = kwargs['unit']
    #     data = data[~data.isnull().values]
    #     ## get unique days from the data
    #     days = kwargs['days']
    #     if len(days)>=1:
    #         d = 1
    #         if unit == 'mmol':
    #             d=18
    #         f = lambda x: 1.509*(np.log(d*x)**1.084-5.381)

    #         fx=f(data)
    #         rfl = lambda x: 10*x**2 if x<0 else 0 
    #         rfh = lambda x: 10*x**2 if x>0 else 0

    #         df = pd.DataFrame(fx.values,index=data.index,columns=['fx'])
    #         df['rl']=df['fx'].apply(rfl)
    #         df['rh']=df['fx'].apply(rfh)

    #         LR = np.zeros(len(days))
    #         HR = np.zeros(len(days))

    #         for i,day in enumerate(days):
    #             rh_data = return_data(df,'rh',day)
    #             rl_data = return_data(df,'rl',day)
    #             LR[i]=max(rl_data)
    #             HR[i]=max(rh_data)
    #         adrr_m = (LR+HR).mean()
    #         adrr_l = LR.mean()
    #         adrr_h = HR.mean()
    #     else:
    #         return np.nan
    #     return adrr_m,adrr_l,adrr_h
    
    # def adrr_easy(self,**kwargs):
    #     """
    #     adrr_easy - returns average daily risk range as calculated using
    #                 the algorithm from easyGV. It differs from the algorithm
    #                 in this calculation because our datetime is used to pull 
    #                 data from each day instead of using the first time as a 
    #                 reference and using the next 24 hours.
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #     Output: ADRR as three values - sum of low and high, low risk rate, high risk rate.
    #     """
    #     data = kwargs['data']
    #     unit = kwargs['unit']
    #     data = data[~data.isnull().values]
    #     d = 1
    #     if unit == 'mmol':
    #         d=18
    #     f = lambda x: 1.509*(np.log(d*x)**1.084-5.381)
        
    #     fx=f(data)
    #     rfl = lambda x: 10*x**2 if x<0 else 0 
    #     rfh = lambda x: 10*x**2 if x>0 else 0
    #     df = pd.DataFrame(fx.values,index=data.index,columns=['fx'])
    #     df['rl']=df['fx'].apply(rfl)
    #     df['rh']=df['fx'].apply(rfh)
    #     LR = []
    #     HR = []
    #     for i in range(len(data)//288+1):
    #         d = df.iloc[i*288:(i+1)*288]
    #         LR.append(d['rl'].max())
    #         HR.append(d['rh'].max())
    #     LR = np.array(LR)
    #     HR = np.array(HR)
    #     return (LR+HR).mean(),LR.mean(),HR.mean()
    
    # def m_value(self,**kwargs):
    #     """
    #     m_value - calculates the M-value for a glucose 
    #               time series. 
    #     Input: data - pandas Series with index as a datetime,
    #                   values are glucose 
    #                   readings associated with those times.
    #            type_ - calculates either the algorithm 
    #                    from the "paper" or "easy"
    #            index - the value used as the index, 
    #                    default is 120
    #            unit - "mg" for milligrams per deciliter 
    #                    or "mmol" for milimoles per
    #                    liter. Default is "mg".
    #     Output:
    #         M-value as a float or None if type_ is not correct.

    #     """
    #     data = kwargs['data']
    #     type_ = kwargs['type']
    #     unit = kwargs['unit']
    #     index = kwargs['m_index']
    #     data = data[~data.isnull().values]
    #     if unit == 'mmol':
    #         data = 18*data
    #     m_star_abs = np.abs((10*np.log10(data/index))**3)
    #     w = (data.max()-data.min())/20
    #     if type_=='paper':
    #         return m_star_abs.mean()+w
    #     if type_=='easy':
    #         return m_star_abs.mean()
    #     return None
    
    # def glucose_management_indicator(self,**kwargs):
    #     """
    #     glucose_management_indicator - Bergenstal (2018), formerly 
    #       referred to as eA1C, or estimated A1C which is a measure 
    #       converting mean glucose from CGM data to an estimated 
    #       A1C using data from a population and regression.
            
    #     Input: data - pandas Series with index as a datetime, 
    #            values are glucose readings associated with those times.
    #            unit - "mg" for milligrams per deciliter or "mmol" 
    #            for milimoles per
    #                   liter. Default is "mg".
    #     """
    #     data = kwargs['data']
    #     unit = kwargs['unit']
    #     data = data[~data.isnull().values]
    #     if unit == 'mmol':
    #         data = 18*data
    #     if unit == 'mg':
    #         return 3.31+0.02392*data.mean()
    #     return None
    
    # def interquartile_range(self,**kwargs):
    #     """
    #     IQR - inter-quartile range 75th percentile - 25th percentile. 
    #       Danne (2017) had this calculation in one of the figures. 
    #     """
    #     data =kwargs["data"]
    #     unit = kwargs["unit"]
    #     data = data[~data.isnull().values]
    #     if unit == 'mmol':
    #         data = 18*data
    #     q75,q25 = np.percentile(data.values,[75,25])
    #     return q75-q25
    
    # def glycemic_risk_index(self,**kwargs):
    #     """
    #     Glycemic Risk Indicator - (Klonoff 2023)
    #         This risk index is a three number and letter result which represents a composite metric for
    #         the quality of the glycemia from a CGM. 
            
    #     Input - time in range vector representing [x1,x2,n,y2,y1] the percent time in each category
    #             [g<54,54<=g<70,70<=g<180,180<g<250,g>250]
    #     """
    #     #data = kwargs['data']
    #     tir = self.time_in_range(**kwargs)
    #     tir = np.round(tir*100,1)
    #     x1,x2,_,y2,y1 = tir
    #     f = lambda x1,x2:x1+0.8*x2
    #     g = lambda y1,y2:y1+0.5*y2
    #     h = lambda x1,x2,y1,y2: 3*f(x1,x2)+1.6*g(y1,y2)
    #     x = f(x1,x2)
    #     y = g(y1,y2)
    #     gri = h(x1,x2,y1,y2)
        
    #     return gri,x,y
        
        
    
    # def auc(self,**kwargs):
    #     """
    #     auc - area under the curve with a threshold value - converts to a 
    #         glucose*min/day value for area above or below the threshold value.
            
    #     Input: data - pandas Series with index as a datetime, values are glucose 
    #                   readings associated with those times.
    #            thresh - a value for the threshold for the calculation above or below the val.
    #                     default value = 100
    #            above - boolean to calculate above or below the threshold.
    #     """
    #     data = kwargs['data']
    #     thresh = kwargs['thresh']
    #     above = kwargs['above']
        
    #     data = data[~data.isnull().values]
    #     ans = 0
    #     timediff = (data.index[-1]-data.index[0])
    #     total_minutes = (timediff.seconds)/60+(timediff.days)*24*60
    #     if above:
    #         for i in range(len(data)-1):
    #             d1 = data.iloc[i]
    #             d2 = data.iloc[i+1]
    #             if d1>=thresh and d2>=thresh:
    #                 ## dt in minutes
    #                 dt = (data.index[i+1]-data.index[i]).seconds//60
    #                 ans += ((d1-thresh)+(d2-thresh))/2*dt
    #                 ## from paper, overly complicated is equivalent to above eqn
    #                 #ans2 += ((min(d1-thresh,d2-thresh)*dt)+abs(d2-d1)*(dt/2))
    #         return ans/(total_minutes/(60*24))
    #     else:
    #         for i in range(len(data)-1):
    #             d1 = data.iloc[i]
    #             d2 = data.iloc[i+1]
    #             if d1<=thresh and d2<=thresh:
    #                 dt = (data.index[i+1]-data.index[i]).seconds//60
    #                 ans += ((thresh-d1)+(thresh-d2))/2*dt
    #         return ans/(total_minutes/(60*24))
    #     return None
    
    # def auc_5(self,day=True):
    #     """
    #     this function is used to calculate auc above thresh for multi-day
    #         times that are not continuous (i.e. for times between midnight and 6am)
    #         and assumes 5 minutes between all times.
            
    #     thresh - is the value that is currently in the object for the threshold value
    #     """
    #     thresh = self.params['thresh']
    #     if day=='day':
    #         data = self.params['day_data']
    #         num_hours = 18
    #     elif day == "night":
    #         data = self.params['night_data']
    #         num_hours = 6
    #     elif day == "all":
    #         data = self.params['data']
    #         num_hours=24
    #     data=data[~data.isnull().values].values
    #     total_minutes = 5*len(data)
    #     ans = 0
    #     for i in range(len(data)-1):
    #         d1 = data[i]
    #         d2 = data[i+1]
    #         if d1>=thresh and d2>=thresh:
    #             dt=5
    #             ans+=((d1-thresh)+(d2-thresh))/2 * dt
    #     total_days = total_minutes/60
    #     return int(ans/60/total_days)
    
    # def glucose_mean(self,**kwargs):
    #     data = kwargs['data']
    #     return data.mean()
    
    # def glucose_median(self,**kwargs):
    #     data = kwargs['data']
    #     return data.median()
    
    # def glucose_std(self,**kwargs):
    #     data = kwargs['data']
    #     return data.std()
    # def glucose_cv(self,**kwargs):
    #     data = kwargs['data']
    #     return data.std()/data.mean()
    # def glucose_N(self,**kwargs):
    #     data = kwargs['data']
    #     return len(data.dropna())
    
    # def percent_active(self):
    #     data = self.return_period_data(False)
    #     nas = len(data[data.isnull()])
    #     return 1-nas/self.N
    
    ##############################################################
    ####### Statistic Functions ##################################
    ##############################################################
    
    def assign_functions(self):
        
        self.stats_functions = {}
        
        self.stats_functions['num_obs'] = {}
        self.stats_functions['num_obs']['f'] = glucose_N
        self.stats_functions['num_obs']['description']="N"
        self.stats_functions['num_obs']['normal']=[]
        
        self.stats_functions['mean']={}
        self.stats_functions['mean']['f'] = glucose_mean
        self.stats_functions['mean']['description']="Mean"
        self.stats_functions['mean']['normal']=[] #fasting glucose <100
        
        self.stats_functions['std']={}
        self.stats_functions['std']['f'] = glucose_std
        self.stats_functions['std']['description']="STD"
        #2017 Internation Consensus Statement - Hill 2011 => [0,3] 
        self.stats_functions['std']['normal']=[10,26] 
        
        self.stats_functions['cv'] = {}
        self.stats_functions['cv']['f'] = glucose_cv
        self.stats_functions['cv']['description']="CV"
        #2017 Internation Consensus Statement
        self.stats_functions['cv']['normal']=[0,0.36]  
        
        self.stats_functions['mag']={}
        self.stats_functions['mag']['f']=mean_absolute_glucose
        self.stats_functions['mag']['description']="MAG"
        self.stats_functions['mag']['normal']=[0.5,2.2] #Hill 2011
        
        self.stats_functions['tir']={}
        self.stats_functions['tir']['f']=time_in_range
        self.stats_functions['tir']['description']="TIR"
        ## Normal Time in Range 0% less than 54, 4% below 70, >70% of the time
        ## between 70-180, <25% of the time greater than 180, 0% time greater than 250
        self.stats_functions['tir']['normal']=[0,4,70,25,0] #Cleveland Clinic website
        
        self.stats_functions['j_index']={}
        self.stats_functions['j_index']['f']=j_index
        self.stats_functions['j_index']['description']='J_Index'
        self.stats_functions['j_index']['normal']=[4.7,23.6] #Hill 2011
        
        self.stats_functions['bgi']={}
        self.stats_functions['bgi']['f']=low_high_blood_glucose_index
        self.stats_functions['bgi']['description'] = 'LBGI_HGBI'
        self.stats_functions['bgi']['normal'] = {'LBGI':[0,6.9],
                                                 'HBGI':[0,7.7]} #Hill 2011
        
        self.stats_functions['grade']={}
        self.stats_functions['grade']['f']=glycemic_risk_assessment_diabetes_equation
        self.stats_functions['grade']['description']='GRADE'
        self.stats_functions['grade']['normal']=[0,4.7] #Hill 2011
        
        self.stats_functions['gvp']={}
        self.stats_functions['gvp']['f']=glycemic_variability_percentage
        self.stats_functions['gvp']['description']='GVP'
        ## 0-20 Minimal, 20-30 Low, 30-50 Moderate, >50 High
        self.stats_functions['gvp']['normal']=[0,20,30,50] #Peyser 2018
        
        self.stats_functions['li']={}
        self.stats_functions['li']['f']=lability_index
        self.stats_functions['li']['description']='Lability_Index'
        self.stats_functions['li']['normal']=[0,4.7] #Hill 2011
        
        self.stats_functions['modd']={}
        self.stats_functions['modd']['f'] = mean_of_daily_differences
        self.stats_functions['modd']['description']='MODD'
        self.stats_functions['modd']['normal']=[0,3.5] #Hill 2011
        
        self.stats_functions['adrr']={}
        self.stats_functions['adrr']['f'] = average_daily_risk_range
        self.stats_functions['adrr']['description']="ADRR"
        self.stats_functions['adrr']['normal']=[0,8.7] #Hill 2011
        
        self.stats_functions['conga']={}
        self.stats_functions['conga']['f'] = conga
        self.stats_functions['conga']['description']='CONGA'
        self.stats_functions['conga']['normal']=[3.6,5.5] #Hill 2011
        
        self.stats_functions['m_value']={}
        self.stats_functions['m_value']['f'] = m_value
        self.stats_functions['m_value']['description']='M_Value'
        self.stats_functions['m_value']['normal']=[0,12.5] #Hill 2011
        
        
        self.stats_functions['gmi']={}
        self.stats_functions['gmi']['f'] = glucose_management_indicator
        self.stats_functions['gmi']['description']='eA1C'
        self.stats_functions['gmi']['normal']=[0,6] #Danne 2017
        
        self.stats_functions['iqr']={}
        self.stats_functions['iqr']['f'] = interquartile_range
        self.stats_functions['iqr']['description']='Inter-quartile range'
        self.stats_functions['iqr']['normal']=[13,29] #Danne 2017        
        
        self.stats_functions['auc']={}
        self.stats_functions['auc']['f']=auc
        self.stats_functions['auc']['description']='AUC'
        self.stats_functions['auc']['normal'] = []
        
        self.stats_functions['mage']={}
        self.stats_functions['mage']['f']=mean_amplitude_of_glycemic_excursions
        self.stats_functions['mage']['description']='MAGE'
        self.stats_functions['mage']['normal'] = []
        
        self.stats_functions['gri'] = {}
        self.stats_functions['gri']['f']=glycemic_risk_index
        self.stats_functions['gri']['description']='glycemic risk index'
        ##ZoneA - 0-20;ZoneB - 20-40;ZoneC - 40-60;ZoneD - 60-80;ZoneE - 80-100## 
        self.stats_functions['gri']['normal'] = [0,20,40,60,80,100]
        
        return None
    
    def set_params(self,params):
        for key,value in params.items():
            self.params[key]=value
        return self.params
        
    def all_stats(self,data,name=""):
        self.set_params({'data':data})
        desc = []
        values = []
        rv = []
        for key in self.stats_functions.keys():
            desc.append(self.stats_functions[key]['description'])
            values.append(self.stats_functions[key]['f'](**self.params))
        for val in values:
            try:
                rv.append(round(val,3))
            except:
                rv.append([round(e,3) for e in val])
        return pd.DataFrame(rv,index=desc,columns=[name])
    
    def stats_by_day(self):
        res = pd.DataFrame()
        glucose = pd.DataFrame(self.all_data(True))
        for day in self.days:
            data = self.return_data(glucose,glucose.columns[0],day)
            res = pd.concat([res,self.all_stats(data,day)],axis=1)
        return res
    
    def overall_stats_dataframe(self):
        index = []
        vals = []
        for k,v in self.stats.items():
            index.append(k)
            try:
                vals.append(round(v,3))
            except:
                try:
                    vals.append([round(e,3) for e in v])
                except:
                    vals.append(v)
                    
        display = pd.DataFrame(vals,index=index,columns=['Overall Data']).T
        display['total_time']=pd.to_timedelta(display['total_time'])
        return display
            
    def calculate_stats(self):
        inp = self.params
        self.stats['total_time']=self.total_time
        self.stats['percent_active']=percent_active(**inp)
        self.stats['mean']=glucose_mean(**inp)
        self.stats['std']=glucose_std(**inp)
        self.stats['num_obs']=glucose_N(**inp)
        self.stats['cv'] = glucose_cv(**inp)
        print("1")
        self.stats['mag']=mean_absolute_glucose(**inp)
        self.stats['tir']=time_in_range(**inp)
        self.stats['j_index'] = j_index(**inp)
        self.stats['glycemic_variability_percentage']=glycemic_variability_percentage(**inp)
        self.stats['lability_index']=lability_index(**inp)
        print("2")
        self.stats['GRADE_'+inp['type']] = glycemic_risk_assessment_diabetes_equation(**inp)
        self.stats['MODD_'+inp['type']]=mean_of_daily_differences(**inp)
        self.stats['ADRR_'+inp['type']]=average_daily_risk_range(**inp)       
        self.stats['blood_glucose_index_'+inp['type']]=low_high_blood_glucose_index(**inp)
        print("3")
        self.stats['conga1_'+inp['type']]=conga(**inp)
        self.stats['M_value_'+inp['type']]=m_value(**inp)
        self.stats['auc_100+']=auc(**inp)
        inp['above']=False
        self.stats['auc_100-']=auc(**inp)
        inp['above']=True
        self.stats['eA1C']=glucose_management_indicator(**inp)
        print("4")
        self.stats['irq']=interquartile_range(**inp)
        self.stats['wake_AUC']=auc_5(day='day',**inp)
        self.stats['sleep_AUC']=auc_5(day='night',**inp)
        self.stats['24hour_AUC']=auc_5(day='all',**inp)
        self.stats['MAGE']=mean_amplitude_of_glycemic_excursions(**inp)
        self.stats['gri']=glycemic_risk_index(**inp)
        
        
        ### ================== Easy ==========================================###
        
        inp['type']='easy'
        self.stats['conga1_'+inp['type']]=conga(**inp)
        self.stats['blood_glucose_index_'+inp['type']]=low_high_blood_glucose_index(**inp)
        self.stats['GRADE_'+inp['type']] = glycemic_risk_assessment_diabetes_equation(**inp)
        self.stats['MODD_'+inp['type']]=mean_of_daily_differences(**inp)
        self.stats['M_value_'+inp['type']]=m_value(**inp)
        self.stats['ADRR_'+inp['type']]=adrr_easy(**inp)
        inp['type']='paper'
        
        return None
    
    
    
    ##############################################################################
    ############### Graphics Functions ###########################################
    ##############################################################################
    
    
    def plot_all(self):
        """
        plot_all is a summary of all of the days in the file.
        """
        plt.style.use('ggplot')
        df = self.df.copy()
        def convert_to_time(minutes):
            time = []
            for m in minutes:
                hh = m//60
                mm = m%60
                time.append(f'{hh:0>2}:{mm:0>2}')
            return time
        alpha1 = 0.95
        alpha2 = 0.75

        means = df.mean(axis=0)
        stds = df.std(axis=0)

        x = np.array(df.columns)
        x_labels = convert_to_time(x)
        
        plotdata = pd.DataFrame()
        plotdata['mean']=means
        plotdata['std']=stds
        plotdata['dof']=(~df.isna()).sum(axis=0)
        plotdata['t1'] = t.ppf(alpha1,plotdata['dof'],0,1)
        plotdata['t2'] = t.ppf(alpha2,plotdata['dof'],0,1)
        plotdata['low1']=plotdata['mean']-plotdata['t1']*plotdata['std']
        plotdata['low2']=plotdata['mean']-plotdata['t2']*plotdata['std']
        plotdata['high2']=plotdata['mean']+plotdata['t2']*plotdata['std']
        plotdata['high1']=plotdata['mean']+plotdata['t1']*plotdata['std']

        cols_to_plot = ['mean','low1','low2','high1','high2']
        data = plotdata[cols_to_plot].copy()
        datalow1=np.array(data['low1'].values,dtype=float)
        datahigh1=np.array(data['high1'].values,dtype=float)
        datalow2 = np.array(data['low2'].values,dtype=float)
        datahigh2=np.array(data['high2'].values,dtype=float)
        
        fig = plt.figure(figsize=(15,8))
        fig.subplots_adjust(wspace=0.1,hspace=0.4)
        ax = plt.subplot2grid((1,9),(0,1),colspan=8,rowspan=1)
        ax1 = plt.subplot2grid((1,9),(0,0),colspan=1,rowspan=1)
        ax.plot(x,data['mean'],color='black',lw=3,zorder=10)
        ax.plot(x,data['low1'],color='goldenrod',lw=1,ls='--',zorder=5)
        ax.plot(x,data['high1'],color='goldenrod',lw=1,ls='--',zorder=5)
        ax.fill_between(x,datalow1,datahigh1,color='goldenrod',alpha=0.1,zorder=5)
        ax.plot(x,data['low2'],color='cadetblue',lw=2,zorder=7)
        ax.plot(x,data['high2'],color='cadetblue',lw=2,zorder=7)
        ax.fill_between(x,datalow2,datahigh2,color='lightblue',zorder=7)
        low_bar = 70;high_bar=180
        
        ax.hlines([low_bar,high_bar],x[0],x[-1],color='red',lw=0.5)
        ax.set_xticks(ticks = x)
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_major_locator(MultipleLocator(36))
        ax.axvspan(x[0],x[72],facecolor='0.3',alpha=0.5,zorder=-100)
        ax.axvspan(x[72],x[-1],facecolor='0.7',alpha=0.5,zorder=-100)
        ax.set_xlim(x[0],x[-1]+1)
        ax.yaxis.set_visible(False)
        ax.set_ylim()
        
        data = self.stats['tir'][1:-1]
        
        ax1.sharey(ax)
        
        ax1.bar(['TimeInRange'],[low_bar],color='firebrick',alpha=0.5)
        ax1.bar(['TimeInRange'],[high_bar-low_bar],bottom=[low_bar],color='green',alpha=0.5)
        ax1.bar(['TimeInRange'],[250-high_bar],bottom=[high_bar],color='firebrick',alpha=0.5)
        
        # xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        
        heights = [(70+ylim[0])/2,(180+70)//2,(ylim[-1]+180)//2]
        
        ax1.annotate('Time in Range',xy=(-.37,ylim[-1]-15))
        for i,d in enumerate(data):
            
            ax1.annotate(f'{d*100:0.1f}%',xy=(-.2,heights[i]))
        ax.set_title(f"Overview of CGM Data from {self.days[0]} through {self.days[-1]}",
                    fontsize=16)
        
        return fig
    
    def ax_non1(self,ax):
        xlim = ax.get_xlim()
        x_range = xlim[1]-xlim[0]
        ylim = ax.get_ylim()
        y_range = ylim[1]-ylim[0]
        y_mid = (ylim[0]+ylim[1])/2
        x_starts = np.linspace(xlim[0],xlim[1],5)
        ## 1st column #############################
        x_ = xlim[0]+0.02*x_range
        y_10 = y_mid+0.15*y_range
        y_11 = y_mid+0.05*y_range
        y_12 = y_mid-0.05*y_range
        y_13 = y_mid-0.15*y_range
        ax.annotate("Average",xy=(x_,y_10))
        ax.annotate("Daily",xy=(x_,y_11))
        ax.annotate("AUC",xy=(x_,y_12))
        ax.annotate("(mg/dL)*h",xy=(x_,y_13))
        
        ## 2nd Column ############################
        x_=x_starts[1]+0.02*x_range
        y_21 = ylim[1]-0.1*y_range
        y_22 = ylim[1]-0.2*y_range
        ax.annotate("Wake",xy=(x_+0.05*x_range,y_21))
        ax.annotate("6am-12am",xy=(x_,y_22))
        ax.annotate(f"{self.stats['wake_AUC']}",xy=(x_,y_11),weight='bold',fontsize=15)
        ax.annotate("89-121 *",xy=(x_+0.03*x_range,y_13),fontsize=8)
        
        
        ## 3rd Column #############################
        x_=x_starts[2]+0.02*x_range
        ax.annotate("Sleep",xy=(x_+0.05*x_range,y_21))
        ax.annotate("12am-6am", xy=(x_,y_22))
        ax.annotate(f"{self.stats['sleep_AUC']}",xy=(x_,y_11),weight='bold',fontsize=15)
        ax.annotate("85-109 *",xy=(x_+0.03*x_range,y_13),fontsize=8)
        
        ## 4th Column #############################
        x_=x_starts[3]+0.02*x_range
        ax.annotate("24 Hours",xy=(x_,y_21))
        ax.annotate(f"{self.stats['24hour_AUC']}",xy=(x_,y_11),weight='bold',fontsize=15)
        ax.annotate("89-113 *",xy=(x_+0.03*x_range,y_13),fontsize=8)
        
        
        ### Bottom ################################
        x_ = xlim[0]+0.2*x_range
        y_ = ylim[0]+0.1*y_range
        ax.annotate("GLUCOSE EXPOSURE CLOSE-UP",xy=(x_,y_),fontsize=12)
        
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha=0.5,
                                linewidth=5,edgecolor='black',facecolor='white')
        ax.add_patch(rect)
        return ax
    
    def ax_non2(self,ax):
        xlim = ax.get_xlim()
        x_range = xlim[1]-xlim[0]
        ylim = ax.get_ylim()
        y_range = ylim[1]-ylim[0]
        y_mid = (ylim[0]+ylim[1])/2
        x_starts = np.linspace(xlim[0],xlim[1],4)
        ## 1st column #############################
        x_ = xlim[0]+0.05*x_range
        y_11 = ylim[1]-0.1*y_range
        y_12 = ylim[1]-0.2*y_range
        y_13 = y_mid+0.05*y_range
        y_14 = y_mid-0.15*y_range
        ax.annotate("IQR",xy = (x_+0.04*x_range,y_11))
        ax.annotate("mg/dL",xy=(x_,y_12))
        ax.annotate(f"{self.stats['irq']:0.1f}",xy=(x_,y_13),weight='bold',fontsize=15)
        ax.annotate("13-29 *",xy=(x_,y_14))
        
        ## 2nd column ############################
        x_=x_starts[1]-0.05*x_range
        y_21 = y_mid+0.2*y_range
        y_22 = y_mid+0.1*y_range
        y_23 = y_mid-0.05*y_range
        ymin = ylim[0]+0.3*y_range
        ymax = ylim[1]-0.1*y_range
        ax.axvline(x=x_, ymin=ymin, ymax=ymax,color='black')
        x_+=0.1*x_range
        ax.annotate("GVP",xy=(x_,y_11))
        gvp = self.stats['glycemic_variability_percentage']
        ax.annotate(f'{gvp:0.2f}',xy=(x_-0.05*x_range,y_12-0.02*y_range),
                    weight='bold',fontsize=15)
        ax.axhline(y=y_21,xmin=x_starts[1],xmax=x_starts[2]-0.1*x_range, color='black')
        ax.annotate("MODD",xy=(x_-0.02*x_range,y_22))
        modd = self.stats['MODD_paper']
        ax.annotate(f'{modd:0.2f}',xy=(x_-0.05*x_range,y_23),
                    weight='bold',fontsize=15)
        ## 3rd Column #############################
        x_=x_starts[2]-0.05*x_range 
        ax.axvline(x=x_, ymin=ymin, ymax=ymax,color='black')
        x_+=0.1*x_range
        ax.annotate("HGBI",xy=(x_,y_11))
        gbi = self.stats['blood_glucose_index_paper']
        ax.annotate(f'{gbi[1]:0.2f}',xy=(x_-0.05*x_range,y_12-0.02*y_range),
                    weight='bold',fontsize=15)
        ax.axhline(y=y_21,xmin=x_starts[2],xmax=xlim[1]-0.1*x_range, color='black')
        ax.annotate("LGBI",xy=(x_,y_22))
        
        ax.annotate(f'{gbi[0]:0.2f}',xy=(x_-0.05*x_range,y_23),
                    weight='bold',fontsize=15)
        
        ##### Bottom ########################
        x_ = xlim[0]+0.2*x_range
        y_ = ylim[0]+0.1*y_range
        ax.annotate("VARIABILITY CLOSE-UP",xy=(x_,y_),fontsize=12)
        
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha=0.5,
                                linewidth=5,edgecolor='black',facecolor='white')
        ax.add_patch(rect)
        return ax
    
    def ax_text(self,params):
        ax = params['ax']
        txt = params['txt']
        line_offset = params['line_offset']
        val_text_offset = params['val_text_offset']
        vals = params['vals']
        norms = params['norms']
        bottom = params['bottom']
        n = len(txt)
        m = len(bottom)
        
        xlim = ax.get_xlim()
        x_mid = (xlim[0]+xlim[1])/2
        x_range = xlim[1]-xlim[0]
        x_starts = np.linspace(xlim[0],xlim[1],n+1)
        
        ylim = ax.get_ylim()
        y_range = ylim[1]-ylim[0]
        y_mid = (ylim[0]+ylim[1])/2
            
        for i in range(n):
            if i != n-1:
                x_ = (x_starts[i]+x_starts[i+1])/2+line_offset[i]*x_range
                if m==n:
                    ymin = 0.05*(y_range)+ylim[0]
                else:
                    ymin = 0.3*(y_range)+ylim[0]
                ax.axvline(x=x_,
                           ymin=ymin,
                           ymax=ylim[1]-0.1*(y_range),
                           color='black')
            x_ = x_starts[i]+0.05*x_range
            x_ += val_text_offset[i]*x_range
            # Values in middle of figure
            try:
                ax.annotate(f'{vals[i]:0.1f}',(x_,y_mid),weight='bold',fontsize=15)
            except:
                ax.annotate(str(vals[i]),(x_,y_mid),weight = 'bold',fontsize=15)
            ax.annotate(norms[i],xy=(x_,y_mid-0.2*y_range),fontsize=8)
            for j in range(len(txt[i])):
                ## text at the top of the figure
                x_ = x_starts[i]+0.05*x_range
                y_ = ylim[1]-0.1*y_range*j
                ax.annotate(txt[i][j],xy=(x_,y_),fontsize=10)
        
        ## Bottom 
        x_starts = np.linspace(xlim[0],xlim[1],m+1)
        for i in range(len(bottom)):
            x_ = x_starts[i]+0.05*x_range
            for j in range(len(bottom[i])):
                y_ = y_mid-(0.32+0.1*j)*y_range #ylim[0]+0.1*y_range
                if len(bottom[i])==1:
                    y_-=0.1
                ax.annotate(bottom[i][j],xy=(x_,y_))
            
                
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha=0.5,
                                linewidth=5,edgecolor='black',facecolor='white')
        ax.add_patch(rect)
        return ax
    
    def plot_agp(self):
        """
        plot_agp - from 2017 International Consensus on use of CGM. This plot is trying to emulate
            Figure 1 in the paper.
            
        Input: Dexcom object
        
        Output: Figure 1
        """
        plt.style.use('ggplot')
        df = self.df.copy()
        def convert_to_time(minutes):
            time = []
            for m in minutes:
                hh = m//60
                mm = m%60
                time.append(f'{hh:0>2}:{mm:0>2}')
            return time
        alpha1 = 0.90
        alpha2 = 0.75

        means = df.mean(axis=0)
        medians = df.median(axis=0)
        stds = df.std(axis=0)

        x = np.array(df.columns)
        x_labels = convert_to_time(x)
        
        plotdata = pd.DataFrame()
        plotdata['mean']=means
        plotdata['median']=medians
        plotdata['std']=stds
        plotdata['dof']=(~df.isna()).sum(axis=0)
        plotdata['t1'] = t.ppf(alpha1,plotdata['dof'],0,1)
        plotdata['t2'] = t.ppf(alpha2,plotdata['dof'],0,1)
        plotdata['low1']=plotdata['mean']-plotdata['t1']*plotdata['std']
        plotdata['low2']=plotdata['mean']-plotdata['t2']*plotdata['std']
        plotdata['high2']=plotdata['mean']+plotdata['t2']*plotdata['std']
        plotdata['high1']=plotdata['mean']+plotdata['t1']*plotdata['std']
        
        cols_to_plot = ['mean','median','low1','low2','high1','high2']
        data = plotdata[cols_to_plot].copy()
        datalow1=np.array(data['low1'].values,dtype=float)
        datahigh1=np.array(data['high1'].values,dtype=float)
        datalow2 = np.array(data['low2'].values,dtype=float)
        datahigh2=np.array(data['high2'].values,dtype=float)
        
        fig = plt.figure(figsize=(15,15))
        fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
        ax = plt.subplot2grid((15,15),(7,2),colspan=13,rowspan=8)
        ax1 = plt.subplot2grid((15,15),(7,0),colspan=2,rowspan=8)
        
        ## Average Glucose / Glycemic Estimate
        vals = [self.stats['mean'],self.stats['eA1C']]
        norms = ['88-116 *', "<6 *"]
        bottom = [['GLUCOSE EXPOSURE']]
        ax2 = plt.subplot2grid((15,15),(0,0),colspan=3,rowspan=3)
        ax2.set_axis_off()
        ax2=self.ax_text({'ax':ax2,
                          'line_offset':[0.2], #offset or None
                          'txt':[["","Avg","Glucose"],["","Glycemic","Estimate"]],
                          'val_text_offset':[0.08,0.1],
                          'vals':vals,
                          'norms':norms,
                          'bottom':bottom})
        
        ## Time In Range Charts
        vals = self.stats['tir']*100
        norms = ['0 *','<4 *','>90 *','<6 *','0 *']
        bottom = [["Level 2"],["Level 1"], ["GLUCOSE", "RANGES"], ["Level 1"],["Level 2"]]
        ax3 = plt.subplot2grid((15,15),(0,3),colspan=7,rowspan=3)
        ax3.set_axis_off()
        ax3=self.ax_text({'ax':ax3,
                      'line_offset':[0.1,0.1,0.14,0.13], #offset or None
                      'txt':[["","Very Low","Below 54","mg/dL"],
                             ["","Low Alert","Below 70","mg/dL"],
                             ["","In Target","Range 70-180","mg/dL"],
                             ["","High Alert"],
                             ["","Very High"]],
                          'val_text_offset':[0.05,0.05,0.05,0.05,0.05],
                          'norms':norms,
                          'vals':vals,
                          'bottom':bottom})
        
        ## Coefficient of Variation / Std
        vals = [self.stats['cv']*100,self.stats['std']]
        norms = ['19.25 *', '10-26 *']
        bottom = [["GLUCOSE VARIABILITY"]]
        ax4 = plt.subplot2grid((15,15),(0,10),colspan=3,rowspan=3)
        ax4.set_axis_off()
        ax4=self.ax_text({'ax':ax4,
                          'line_offset':[0.2], #offset or None
                          'txt':[["","Coefficient","of Variation"],
                                 ["","SD","mg/dL"]],
                          'val_text_offset':[0.1,0.1],
                          'vals':vals,
                          'norms':norms,
                          'bottom':bottom})
        
        ## % Time CGM Active
        vals = [self.stats['percent_active']*100]
        norms = [""]
        bottom = [['DATA','SUFFICIENCY']]
        ax5 = plt.subplot2grid((15,15),(0,13),colspan=2,rowspan=3)
        ax5.set_axis_off()
        ax5=self.ax_text({'ax':ax5,
                          'line_offset':[0.2], #offset or None
                          'txt':[["","% Time CGM","Active"]],
                          'val_text_offset':[0.2],
                          'vals':vals,
                          'norms':norms,
                          'bottom':bottom})
        
        ## Glucose Exposure Closeup - ax_non1
        ax6 = plt.subplot2grid((15,15),(3,0),colspan=5,rowspan=3)
        ax6.set_axis_off()
        ax6 = self.ax_non1(ax6)
        
        ## Variability Closeup - ax_non2
        ax7 = plt.subplot2grid((15,15),(3,5),colspan=4,rowspan=3)
        ax7.set_axis_off()
        ax7 = self.ax_non2(ax7)
        
        
        #colors - [50% color, 90% color, 75%color outer, 75% color inner]
        colors = ['darkgoldenrod','firebrick','cadetblue','lightblue']
        ax.plot(x,data['median'],color=colors[0],lw=3,zorder=10)
        ax.plot(x,data['low1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.plot(x,data['high1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.fill_between(x,datalow1,datahigh1,color=colors[1],alpha=0.1,zorder=5)
        ax.plot(x,data['low2'],color=colors[2],lw=2,zorder=7)
        ax.plot(x,data['high2'],color=colors[2],lw=2,zorder=7)
        ax.fill_between(x,datalow2,datahigh2,color=colors[3],zorder=7)
        low_bar = 70;high_bar=180
        
        ax.hlines([low_bar,high_bar],x[0],x[-1],color='red',lw=0.5)
        ax.set_xticks(ticks = x)
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_major_locator(MultipleLocator(36))
        ax.axvspan(x[0],x[72],facecolor='0.3',alpha=0.5,zorder=-100)
        ax.axvspan(x[72],x[-1],facecolor='0.7',alpha=0.5,zorder=-100)
        ax.set_xlim(x[0],x[-1]+1)
        ax.yaxis.set_visible(False)
        ax.set_ylim()
        
        data = self.stats['tir']
        data = [data[0]+data[1],data[2],data[3]+data[4]]
        
        ax1.sharey(ax)
        
        ax1.bar(['TimeInRange'],[low_bar],color='firebrick',alpha=0.5)
        ax1.bar(['TimeInRange'],[high_bar-low_bar],bottom=[low_bar],color='green',alpha=0.5)
        ax1.bar(['TimeInRange'],[250-high_bar],bottom=[high_bar],color='firebrick',alpha=0.5)
        
        # xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        
        heights = [(70+ylim[0])/2,(180+70)//2,(ylim[-1]+180)//2]
        
        ax1.annotate('Time in Range',xy=(-.37,ylim[-1]-5))
        for i,d in enumerate(data):
            
            ax1.annotate(f'{d*100:0.1f}%',xy=(-.2,heights[i]))
        ax.set_title(f"Overview of CGM Data from {self.days[0]} through {self.days[-1]}",
                    fontsize=16)
        
        plt.show()
        return None
    
    def plot_gri(self):
        """
        plot the glycemic risk index on a chart from the paper.
        """
        points = glycemic_risk_index(**self.params)
        zones = np.array([0,20,40,60,80,100])
        zones_=['A','B','C','D','E']
        
        pt0 = points[0]
        idx = np.where(pt0>=zones)[0][-1]
        zone = zones_[idx]
        
        x = np.linspace(0,30,1000)
        fa = lambda x:(20-3*x)/1.6
        fb = lambda x:(40-3*x)/1.6
        fc = lambda x:(60-3*x)/1.6
        fd = lambda x:(80-3*x)/1.6
        fe = lambda x:(100-3*x)/1.6
        fig,ax = plt.subplots(figsize=(12,12))
        ya = fa(x)
        yb = fb(x)
        yc = fc(x)
        yd = fd(x)
        ye = fe(x)
        ax.plot(x,ya,color="green")
        ax.fill_between(x,0,ya,color='green',alpha=0.3)
        ax.plot(x,yb,color='yellow')
        ax.fill_between(x,ya,yb,color='yellow',alpha=0.3)
        ax.plot(x,yc,color='orange')
        ax.fill_between(x,yb,yc,color='orange',alpha=0.3)
        ax.plot(x,yd,color='orangered')
        ax.fill_between(x,yc,yd,color='orangered',alpha=0.4)
        ax.plot(x,ye,color='darkred')
        ax.fill_between(x,yd,ye,color='darkred',alpha=0.3)
        ax.set_xlim(0,30)
        ax.set_ylim(0,60)
        ax.set_xlabel("Hypoglycemia Component (%)")
        ax.set_ylabel("Hyperglycemia Component (%)")
        
        ax.scatter(points[1],points[2],s=50,color = 'black',marker = 'o')
        ax.annotate(zone,xy=(points[1]+0.5,points[2]+0.5))
        plt.show()