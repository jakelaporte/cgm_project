import streamlit as st
import pandas as pd
import os
from ..cgm_obj_v03 import CGM

class st_CGM(object):
    def __init__(self,names,files,dt_col=1,gl_col=7,dt_fmt='%Y-%m-%dT%H:%M:%S'):
        self.names = names
        self.files = files
        self.data={}
        for name,file in zip(names,files):
            self.data[name]=CGM(file,dt_col,gl_col,dt_fmt)
        self.selected_file = self.names[0]
    def view_data_summary(self):
        names = self.names
        name = st.selectbox("Select the file.",
                              index=names.index(self.selected_file),
                              options = names)
        st.pyplot(self.data[name].plot_agp())
        st.divider()
        st.write(self.data[name].overall_stats_dataframe())
        st.divider()
        st.write(self.data[name].stats_by_day())
        self.selected_file = name
        
    def view_df_series(self):
        names = self.names
        name = st.selectbox("Select the file.",
                            index = names.index(self.selected_file),
                            options = names,
                            key = 'viewdf')
        st.write(self.data[name].df)
        st.divider()
        st.write(self.data[name].series)
        st.divider()
        st.write(self.data[name].periods)
        self.selected_file = name
        
    def view_gri(self):
        names = self.names
        name = st.selectbox("Select the file.",
                            index = names.index(self.selected_file),
                            options = names,
                            key = 'viewgri')
        st.pyplot(self.data[name].plot_gri())
        self.selected_file = name
        
    def export_data(self,filename,path):
        df = pd.DataFrame()
        for name in self.names:
            df = pd.concat([df,self.data[name].overall_stats_dataframe()])
        df['idx']=self.names
        df.set_index('idx',inplace=True)
        st.write(df)
        path = os.path.abspath(path)
        path_file_to_write = os.path.join(path,filename)
        df.to_csv(path_file_to_write)

class FileOperations(object):
    def __init__(self,curr_dir):
        self.home_dir = curr_dir
        self.curr_dir = curr_dir
        self.list_directory()
        return None

    def list_directory(self):
        curr_dir = self.curr_dir
        folder_names = ['..','.']
        file_names = []
        files_folders = os.listdir(curr_dir)
        for ff in files_folders:
            if ('.' not in ff[2:]):
                folder_names.append('./'+ff)
            else:
                file_names.append(ff)
        self.curr_folders = sorted(folder_names)
        self.curr_files = sorted(file_names)
        return None
    
    def display_files(self, dir_):
        dir_ = os.path.join(self.curr_dir,dir_)
        files = os.listdir(dir_)
        res = ""
        res+="| File # | File Name |\n"
        res+="|--------|-----------|\n"
        for i,file in enumerate(files):
            res+=f'|{i+1}|{str(file)}|\n'
        return res
    
    def change_dir(self,new_dir):
        curr_dir = self.curr_dir
        if new_dir == "..":
            curr_dir = "\\".join(curr_dir.split('\\')[:-1])
        elif new_dir[:2]=='./':
            curr_dir = os.path.realpath(os.path.join(curr_dir,new_dir))
        self.curr_dir = curr_dir
        self.list_directory()
        return None
    
    def __str__(self):
        return str(self.curr_dir)

