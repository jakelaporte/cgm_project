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
                        auc_thresh,mean_amplitude_of_glycemic_excursions,
                        glycemic_risk_index,percent_active,auc_5,
                        adrr_easy,)


class CGM(object):
    def __init__(self,filename,dt_col=1,gl_col=7,dt_fmt='%Y-%m-%dT%H:%M:%S',skip_rows=1):
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
        self.series,self.periods = read_data(filename,dt_col=dt_col,
                                             gl_col=gl_col,
                                             dt_fmt = dt_fmt,
                                             skip_rows=skip_rows)
        self.dt_fmt = dt_fmt
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
        self.daily_stats = None
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
        self.stats_functions['auc']['f']=auc_thresh
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
        if self.daily_stats is None:
            res = pd.DataFrame()
            glucose = self.params['series'][['imputed']]
            for day in self.days:
                data = return_data(glucose,glucose.columns[0],day)
                res = pd.concat([res,self.all_stats(data,day)],axis=1)
            self.set_params({'data':return_period_data(self.series,self.periods,True)})
            self.daily_stats = res
        else:
            res = self.daily_stats
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
        self.stats['auc_100+']=auc_thresh(**inp)
        inp['above']=False
        self.stats['auc_100-']=auc_thresh(**inp)
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
        return fig
    
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
        return (fig)