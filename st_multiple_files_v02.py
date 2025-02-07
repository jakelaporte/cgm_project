# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:12:07 2024

@author: grover.laporte
"""

import streamlit as st
from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tempfile

from cgm_object import *

st.title("CGM Application")
if 'cgm_data' not in st.session_state:
    st.session_state['cgm_data'] = None
    
if 'test_file' not in st.session_state:
    st.session_state['test_file'] = None
    
if 'dt_col' not in st.session_state:
    st.session_state['dt_col'] = None
    
if 'gl_col' not in st.session_state:
    st.session_state['gl_col'] = None

if 'dt_fmt' not in st.session_state:
    st.session_state['dt_fmt']= '%Y-%m-%dT%H:%M:%S'

if 'datetime_fmt' not in st.session_state:
    st.session_state['datetime_fmt'] = None
    
if 'skip_rows' not in st.session_state:
    st.session_state['skip_rows'] = 1
    
if 'imported_files' not in st.session_state:
    st.session_state['imported_files'] = []

if 'imported_file_names' not in st.session_state:
    st.session_state['imported_file_names']=[]
    
if 'stats' not in st.session_state:
    st.session_state['stats'] = None
    
    
def clear_session():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    if 'cgm_data' not in st.session_state:
        st.session_state['cgm_data'] = None
        
    if 'test_file' not in st.session_state:
        st.session_state['test_file'] = None
        
    if 'dt_col' not in st.session_state:
        st.session_state['dt_col'] = None
        
    if 'gl_col' not in st.session_state:
        st.session_state['gl_col'] = None

    if 'dt_fmt' not in st.session_state:
        st.session_state['dt_fmt']= '%Y-%m-%dT%H:%M:%S'

    if 'datetime_fmt' not in st.session_state:
        st.session_state['datetime_fmt'] = None
        
    if 'imported_files' not in st.session_state:
        st.session_state['imported_files'] = []

    if 'imported_file_names' not in st.session_state:
        st.session_state['imported_file_names']=[]
        
def change_attribute(key,val):
    val = st.session_state['skip_key']
    st.session_state[key]=val

    

options = [":open_file_folder: Files",
           ":man: Individual Analysis",
           ":woman-woman-boy-boy: Cohort Analysis",
           ":floppy_disk: Export Data"]

select = st.sidebar.radio(label = "Select the tool:",
                      options = options,
                      key='sb_select')

if select == options[0]:
    
    tab0,tab1,tab2 = st.tabs(["Instructions",
                                  "Import Data",
                                   "Datetime"])
    
    with tab0:
        st.subheader("Instructions")
        #body = "[Instructions](https://youtu.be/t17dbV0pGh4)\n\n"
        #st.video('https://youtu.be/t17dbV0pGh4',format="video/mp4")
        body=''
        body+=":red[This application takes `.csv` files with CGM data "
        body+="as input and produces variability metrics for that data along "
        body+="with graphs to visualize the data.\n It is important that "
        body+="the data used has the same structure in terms of columns. The "
        body+="`Import Data` section of this tool will use the column number "
        body+="of the glucose and the date-time of your data "
        body+="selected below. Be sure to select these columns based on "
        body+="your data.]  \n\n"
        body+="#### Video: [Getting Started](https://youtu.be/S_QklfS5XCw)"
        st.markdown(body)
        st.divider()
        test_file = st.file_uploader("Select file to explore.",
                                     type = 'csv')
        skip_rows = st.session_state['skip_rows']
        if test_file is not None:
            raw_file = deepcopy(test_file)
            df = view_raw_data(raw_file.read(),skip_rows)
            st.write(df.head(5))
            
            st.divider()
            ## Determine rows to delete
            skip_rows = st.number_input("Number of rows to skip:",
                                        min_value = 1,max_value = 10,
                                        value="min",step=1,
                                        on_change=change_attribute,
                                        key = 'skip_key',
                                        args=('skip_rows',1))
            
            ## Determine column numbers
            cols = st.columns(2)
            with cols[0]:
                st.markdown("##### Select Date-Time Column:")
                time_date_df = pd.DataFrame({'columns':list(df.columns),
                                          'date_time':[False]*len(df.columns)})
                time_date_df = st.data_editor(time_date_df,hide_index=True)
                
            with cols[1]:
                st.markdown("##### Select Glucose Column:")
                glucose_df = pd.DataFrame({'columns':list(df.columns),
                                          'glucose':[False]*len(df.columns)}) 
                glucose_df = st.data_editor(glucose_df,hide_index=True)
                
            # Use the data_editor for selecting the columns for glucose / date-time
            dt_col = np.where(time_date_df.values[:,1]==True)
            gl_col = np.where(glucose_df.values[:,1] == True)
            fmt_str = st.session_state['dt_fmt']
            try:
                dt_col = dt_col[0][0]; gl_col = gl_col[0][0]
                cols = df.columns[[dt_col,gl_col]]
                #df[cols[0]] = pd.to_datetime(format='%Y-%m-%dT%H:%M:%S')
                try:
                    df[cols[0]]=(pd.to_datetime(df[cols[0]],format=fmt_str))
                except:
                    st.markdown("##### Video [Timestamp](https://youtu.be/gW4RNwWobi4)")
                    fmt_str = st.text_input("Date-Time Format:",
                                            value=fmt_str)
                    df[cols[0]]=(pd.to_datetime(df[cols[0]],format=fmt_str))
                    
                st.write("If this is correct, continue to `Import Data`.")
                st.write(df[cols])
                st.session_state['dt_col']=dt_col
                st.session_state['gl_col']=gl_col
                st.session_state['dt_fmt']=fmt_str
            except:
                pass

    with tab1:
        cgm_data = st.session_state['cgm_data']
        dt_col = st.session_state['dt_col']
        gl_col = st.session_state['gl_col']
        dt_fmt = st.session_state['dt_fmt']
        skip_rows = st.session_state['skip_rows']
        st.markdown("#### Import csv file [Upload Files](https://youtu.be/xYmMLfRtc4E)")
        if (cgm_data is None): #and (dt_col is not None):
            files = []
            names = []
            uploaded_files = st.file_uploader("Select .csv files to upload.",
                                             type='csv',
                                             accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                names.append(uploaded_file.name)
                files.append(uploaded_file.read())
            if len(files)>0:
                cgm_data = st_CGM(names,files,dt_col,gl_col,dt_fmt,skip_rows)
                st.session_state['cgm_data'] = cgm_data
        else:
            st.write("Click the reload button to start a new analysis.")
            st.button("Reload",on_click=clear_session)
    with tab2:
        pass
        # st.subheader("Correct a datetime.")
        # cgm_data = st.session_state['cgm_data']
        # dt_col = st.session_state['dt_col']
        # gl_col = st.session_state['gl_col']
        # dt_fmt = st.session_state['dt_fmt']
        # imported_files = st.session_state['imported_files']
        # start_date = ""
        # body = ""
        # body+=":red[The files selected in this tab should all have the same "
        # body+="structure in terms of datetime column, format and glucose "
        # body+="column. This app will change the datetime format to be "
        # body+="consistent with the requirements of this app.]\n\n"
        # body+=":blue[For example, if a datetime column only has a time, this "
        # body+="tab will help change the data to a datetime in order to "
        # body+="function with the rest of the Import function, Data Tools, and "
        # body+="Export Data.]"
        # st.markdown(body)
        # st.divider()

        # if len(imported_files)==0:
        #     dt_files = []
        #     dt_names = []
        #     imported_files = st.file_uploader('Select .csv files to change datetime.',
        #                                  type='csv',
        #                                  accept_multiple_files=True)
        #     for imported_file in imported_files:
        #         dt_files.append(imported_file.read())
        #         dt_names.append(imported_file.name)
        #     st.session_state['imported_files']=dt_files
        #     st.session_state['imported_file_names']=dt_names
        # if len(imported_files)>0:
        #     dt_files = st.session_state['imported_files']
        #     dt_names = st.session_state['imported_file_names']
        #     name = st.selectbox("Select a file to inspect.",
        #                         index=0,
        #                         options = dt_names)
        #     idx = dt_names.index(name)
        #     cols = st.columns(2)
        #     with cols[0]:
        #         st.write("Current datetime and glucose values.")
        #         view_data(dt_files[idx],dt_col,gl_col,dt_fmt)
        #     with cols[1]:
        #         st.write("Transformed datetime:")
        #         include_start_date = st.checkbox("Start Date Needed")
        #         if include_start_date:
        #             start_date = st.date_input("Start Date",
        #                                        format='YYYY-MM-DD')
        #             st.write("Format for YYYY-MM-DD = `%Y-%m-%d` ")
        #             if len(dt_fmt)<10:
        #                 dt_fmt = '%Y-%m-%d '+dt_fmt
                        
        #         dt_fmt=st.text_input("Change datetime format:",value=dt_fmt)
        #         try:
        #             data = transform_data(dt_files[idx],dt_col,gl_col,
        #                                   dt_fmt,start_date)
        #             st.write(data)
        #         except:
        #             view_data(dt_files[idx],dt_col,gl_col,dt_fmt)
                    
            
if select == options[1]:
    pill_options = ["View Data","Data Summary","Glycemic Risk Index"]
    selection = st.pills("Select a tool:",
                         options = pill_options,
                         default = pill_options[0])
    cgm_data = st.session_state['cgm_data']
    con1 = st.container(border=True)
    con2 = st.container(border=True)
    if cgm_data is not None:
        names = cgm_data.names
        name = con1.selectbox("Select the file.",
                      index = names.index(cgm_data.selected_file),
                      options = names)
        cgm_data.selected_file = name
        
        if selection == pill_options[0]:
            with con2:
                cgm_data.view_data_summary(name)
        if selection == pill_options[1]:
            with con2:
                cgm_data.view_df_series(name)
        if selection == pill_options[2]:
            with con2:
                cgm_data.view_gri(name)
                
if select == options[2]:
    cohort_options = ['Select Stats','Correlation','Visual Comparison']
    selection = st.pills("",
                         options = cohort_options,
                         default = cohort_options[0],
                         key = 'cohort_pills')
    cgm_data = st.session_state['cgm_data']
    stats = st.session_state['stats']
    con3 = st.container(border=True)
    con4 = st.container(border=True)
    include = st.session_state['stats']
    if cgm_data is not None:
        if selection == cohort_options[0]:
            stats = list(cgm_data.df.columns)
            st.multiselect("Choose the stats for the analysis.",
                                     options = stats,
                                     default = include,
                                     key = 'include')
            
        if selection == cohort_options[1]:
            try:
                stats = st.session_state['include']
            except:
                pass
            data = cgm_data.df[stats].copy()
            data.index = np.arange(len(data))
            corr = data[stats].corr()
            st.write(corr)
            fig = plt.figure()
            sns.heatmap(corr)
            st.pyplot(fig)
            st.session_state['stats'] = stats
            
        if selection == cohort_options[2]:
            stats = st.session_state['stats']
            if stats is not None:
                compare=st.multiselect("Select two statistics to compare.",
                               options = stats,
                               key='compare',
                               max_selections=2)
                if len(compare)>1:
                    fig = plt.figure()
                    sns.scatterplot(x=compare[0],y=compare[1],
                                    data=cgm_data.df[compare])
                    st.pyplot(fig)
        
        
if select == options[3]:
    tab1, tab2 = st.tabs(["Export Stats","Export Future"])
    with tab2:
        pass
    
    with tab1:
        cgm_data = st.session_state['cgm_data']
        st.subheader("Export Data")
        filename = st.text_input("Name of File",'cgm_download.csv')
        ok = st.button("OK")
        if ok:
            cgm_data.export_data(filename)
    
    
    
    
