#%% Import Modules
import sys
import os
# Add the directory of the current script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from functions import prepare_juror_lists, create_weights, create_nested_tuples
import pandas as pd

#%%Specify key variables
data_fn: str = "12-18-24_ZestFG_Data"+".xlsx" #Filename for pre-stim / CAS dataset
data_sheetName: str = "values" #Use Labeled Data Sheet
juror_varName: str = "Name" #Name of juror ID variable
dv: str = "DV_1PL_0Def" #Name of verdict DV
pl_label = 1 #Label for Plaintiff jurors in DV
def_label = 0 #Label for Defense jurors in DV

chisq_fn: str = "Zest_ChiSqResults"+".xlsx" #Filename for chi-square output file
chisq_sheetName: str = "pruned"

#%%Import data files
data_df = pd.read_excel(os.path.join(script_dir,"Data", data_fn), sheet_name = data_sheetName)
data_df = data_df.drop(columns = ["wedge"])

jurors, pl_jurors, def_jurors = prepare_juror_lists(data_df, dv, juror_varName, pl_label, def_label)

chisq_df = pd.read_excel(os.path.join(script_dir,"Data", chisq_fn), sheet_name = chisq_sheetName)

# # Apply the function to create the combined tuple columns
# chisq_df['iv_tuple'] = chisq_df.apply(
#     lambda row: create_tuple(row, ['control_2', 'control_1', 'iv']), axis=1
# )

# chisq_df['level_tuple'] = chisq_df.apply(
#     lambda row: create_tuple(row, ['control_2_level', 'control_1_level', 'iv_level']), axis=1
# )

# chisq_df['label_tuple'] = chisq_df.apply(
#     lambda row: create_tuple(row, ['control_2_label', 'control_1_label', 'iv_label']), axis=1
# )

chisq_df['iv_level_tuples'] = chisq_df.apply(create_nested_tuples, axis=1)


# for i, row in chisq_df.iterrows():
#     jurors_by_iv = [getattr(juror, dv) for jNum,juror in jurors.items() if getattr(juror, row['iv']) == row['iv_level']]
#     print(jurors_by_iv)

#%%

plf_predictions_df = create_weights(chisq_df, .59)

plf_predictions_df.to_excel(os.path.join(script_dir, "Output", "Zest_weighted_PLF_predictions.xlsx"), index=False)
