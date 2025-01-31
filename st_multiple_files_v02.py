# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:12:07 2024

@author: grover.laporte
"""

import streamlit as st
from copy import deepcopy
import numpy as np
import pandas as pd
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
    
if 'imported_files' not in st.session_state:
    st.session_state['imported_files'] = []

if 'imported_file_names' not in st.session_state:
    st.session_state['imported_file_names']=[]
    
if 'path_name' not in st.session_state:
    st.session_state['path_name']=None

    
options = [":open_file_folder: Files",
           ":desktop_computer: Data Tools",
           ":floppy_disk: Export Data"]

select = st.sidebar.radio(label = "Select the tool:",
                      options = options,
                      key='sb_select')

if select == options[0]:
    
    tab0,tab3,tab1,tab2 = st.tabs(["Instructions",
                                  "Select Output Folder",     
                                  "Import Data",
                                   "Datetime"])
    
    with tab0:
        st.subheader("Instructions")
        body = ""
        body+=":red[This application takes `.csv` files with CGM data "
        body+="as input and produces variability metrics for that data along "
        body+="with graphs to visualize the data.\n It is important that "
        body+="the data used has the same structure in terms of columns. The "
        body+="`Import Data` section of this tool will use the column number "
        body+="of the glucose and the date-time of your data "
        body+="selected below. Be sure to select these columns based on "
        body+="your data.]"
        st.markdown(body)
        st.divider()
        test_file = st.file_uploader("Select file to explore.",
                                     type = 'csv')
        if test_file is not None:
            raw_file = deepcopy(test_file)
            df = pd.read_csv(test_file)
            st.write(df.head(5))
            st.write(view_raw_data(raw_file.read()))
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
        st.subheader("Import csv file")
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
                cgm_data = st_CGM(names,files,dt_col,gl_col,dt_fmt)
                st.session_state['cgm_data'] = cgm_data       
    with tab2:
        st.subheader("Correct a datetime.")
        cgm_data = st.session_state['cgm_data']
        dt_col = st.session_state['dt_col']
        gl_col = st.session_state['gl_col']
        dt_fmt = st.session_state['dt_fmt']
        imported_files = st.session_state['imported_files']
        folder = st.session_state['path_name']
        start_date = ""
        body = ""
        body+=":red[The files selected in this tab should all have the same "
        body+="structure in terms of datetime column, format and glucose "
        body+="column. This app will change the datetime format to be "
        body+="consistent with the requirements of this app.]\n\n"
        body+=":blue[For example, if a datetime column only has a time, this "
        body+="tab will help change the data to a datetime in order to "
        body+="function with the rest of the Import function, Data Tools, and "
        body+="Export Data.]"
        st.markdown(body)
        st.divider()

        if len(imported_files)==0:
            dt_files = []
            dt_names = []
            imported_files = st.file_uploader('Select .csv files to change datetime.',
                                         type='csv',
                                         accept_multiple_files=True)
            for imported_file in imported_files:
                dt_files.append(imported_file.read())
                dt_names.append(imported_file.name)
            st.session_state['imported_files']=dt_files
            st.session_state['imported_file_names']=dt_names
        if len(imported_files)>0:
            dt_files = st.session_state['imported_files']
            dt_names = st.session_state['imported_file_names']
            name = st.selectbox("Select a file to inspect.",
                                index=0,
                                options = dt_names)
            idx = dt_names.index(name)
            cols = st.columns(2)
            with cols[0]:
                st.write("Current datetime and glucose values.")
                view_data(dt_files[idx],dt_col,gl_col,dt_fmt)
            with cols[1]:
                st.write("Transformed datetime:")
                include_start_date = st.checkbox("Start Date Needed")
                if include_start_date:
                    start_date = st.date_input("Start Date",
                                               format='YYYY-MM-DD')
                    st.write("Format for YYYY-MM-DD = `%Y-%m-%d` ")
                    if len(dt_fmt)<10:
                        dt_fmt = '%Y-%m-%d '+dt_fmt
                        
                dt_fmt=st.text_input("Change datetime format:",value=dt_fmt)
                try:
                    data = transform_data(dt_files[idx],dt_col,gl_col,
                                          dt_fmt,start_date)
                    st.write(data)
                except:
                    view_data(dt_files[idx],dt_col,gl_col,dt_fmt)
                    
                    
    with tab3:
        ### Path Selection ###
        st.divider()
        
        st.markdown("#### Select a folder to download tranformed files:")
        cols = st.columns(2)
        if folder is None:
            folder = FileOperations(os.getcwd())
        with cols[0]:
            st.write(":open_file_folder: Select a Folder:")
            selection = st.radio(" ",options = folder.curr_folders)
            explore = st.button("Explore",key='explore_btn')
        with cols[1]:
            st.write(":file_folder: Files in selected folder:")
            st.markdown(folder.display_files(selection))
            
        st.divider()
        ok_btn = st.button("OK", key='ok_btn')
        st.markdown('###### Current download folder')
        if folder is not None:
            st.write(folder)
        if explore:
            folder.change_dir(selection)
            st.write(folder.curr_dir)
            st.write(folder.curr_folders)
            st.session_state['path_name']=folder
            st.rerun()
        if ok_btn:
            st.session_state['path_name']=folder
            
            
        
if select == options[1]:
    tab1,tab2,tab3 = st.tabs(["View Data",
                         "Data Summary",
                         "Glycemic Risk Index"])
    
    with tab1:
        cgm_data = st.session_state['cgm_data']
        if cgm_data is not None:
            cgm_data.view_data_summary()
    
    with tab2:
        cgm_data = st.session_state['cgm_data']
        if cgm_data is not None:
            cgm_data.view_df_series()
    with tab3:
        cgm_data = st.session_state['cgm_data']
        if cgm_data is not None:
            cgm_data.view_gri()
    
        
if select == options[2]:
    tab1, tab2 = st.tabs(["Export Folder", "Export Data"])
    with tab1:
        pass
    
    with tab2:
        cgm_data = st.session_state['cgm_data']
        folder = st.session_state['path_name']
        st.subheader("Export Data")
        filename = st.text_input("Name of File","download.csv")
        ok = st.button("OK")
        if ok:
            if folder is not None:
                cgm_data.export_data(filename,folder.curr_dir)
            else:
                cgm_data.export_data(filename,os.getcwd())
    
    
    
    
