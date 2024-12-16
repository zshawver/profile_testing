#%% Import Modules
import sys
import os
# Add the directory of the current script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from functions import prepare_juror_lists
import pandas as pd

#%%Specify key variables
data_fn: str = "12-9-24_Gerber_PreStim"+".xlsx" #Filename for pre-stim / CAS dataset
data_sheetName: str = "use_data_quant" #Use Labeled Data Sheet
juror_varName: str = "NAME" #Name of juror ID variable
dv: str = "Lean" #Name of verdict DV
pl_label = 1 #Label for Plaintiff jurors in DV
def_label = 0 #Label for Defense jurors in DV

chisq_fn: str = "Gerber_main_effects"+".xlsx" #Filename for chi-square output file
chisq_sheetName: str = "Main Effect - Labels"

#%%Import data files
data_df = pd.read_excel(os.path.join(script_dir,"Data", data_fn), sheet_name = data_sheetName)

jurors, pl_jurors, def_jurors = prepare_juror_lists(data_df, dv, juror_varName, pl_label, def_label)

chisq_df = pd.read_excel(os.path.join(script_dir,"Data", chisq_fn), sheet_name = chisq_sheetName)

chisq_df.iloc[1]

dietJurors = [juror for jNum, juror in jurors.items() if getattr(juror,'DIET_OR_LIFESTYLES') == 1 ]

for i, row in chisq_df.iterrows():
    jurors_by_iv = [juror.Lean for jNum,juror in jurors.items() if getattr(juror, row['iv']) == row['iv_level']]
    print(jurors_by_iv)
