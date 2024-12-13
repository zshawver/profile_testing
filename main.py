#%% Import Modules
import sys
import os
# Add the directory of the current script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from functions import prepare_juror_lists
import pandas as pd

#%%Specify key variables
data_fn: str = ""+".xlsx" #Filename for pre-stim / CAS dataset
data_sheetName: str = "" #Use Labeled Data Sheet
juror_varName: str = "" #Name of juror ID variable
dv: str = "" #Name of verdict DV
pl_label: str = "" #Label for Plaintiff jurors in DV
def_label: str = "" #Label for Defense jurors in DV

#%%Import data files
data_df = pd.read_excel(os.path.join(script_dir,"Data", data_fn), sheet_name = "data_sheetName")

jurors, pl_jurors, def_jurors = prepare_juror_lists(data_df, dv, juror_varName, pl_label, def_label)
