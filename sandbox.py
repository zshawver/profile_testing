

# # Example juror_data
# juror_data = pd.DataFrame({
#     'juror_name': ['John S','Bob R','Mike T','James W','Will F','Carl W','Tina T','Laura L','Candice B','Amy P','Rachel D','Seth M','Adam S','Jason B','Kristen W'],
#     'DV_1PL_0Def': ['Defense','Plaintiff','Plaintiff','Defense','Defense','Plaintiff','Defense','Plaintiff','Defense','Plaintiff','Defense','Defense','Plaintiff','Defense','Plaintiff'],
#     'GENDER': ['Gender: Man','Gender: Woman','Gender: Man','Gender: Woman','Gender: Man','Gender: Man','Gender: Man','Gender: Man','Gender: Woman','Gender: Man','Gender: Woman','Gender: Woman','Gender: Woman','Gender: Man','Gender: Woman'],
#     'RACE': ['Race: White / Caucasian','Race: White / Caucasian','Race: White / Caucasian','Race: Black / African American','Race: White / Caucasian','Race: Hispanic / Latinx','Race: White / Caucasian','Race: Hispanic / Latinx','Race: Black / African American','Race: Black / African American','Race: White / Caucasian','Race: White / Caucasian','Race: Black / African American','Race: Black / African American','Race: Black / African American'],
#     'PA': ['Political Affiliation: Democrat','Political Affiliation: Not Registered / Unaffiliated / Other','Political Affiliation: Democrat','Political Affiliation: Not Registered / Unaffiliated / Other','Political Affiliation: Democrat','Political Affiliation: Independent','Political Affiliation: Democrat','Political Affiliation: Republican','Political Affiliation: Democrat','Political Affiliation: Democrat','Political Affiliation: Republican','Political Affiliation: Democrat','Political Affiliation: Democrat','Political Affiliation: Independent','Political Affiliation: Democrat'],
#     'AGE_RANGE': ['Age: 50-59','Age: 30-39','Age: 40-49','Age: 20-29','Age: 60-69','Age: 30-39','Age: 60-69','Age: 50-59','Age: 40-49','Age: 60-69','Age: 60-69','Age: 70+','Age: 60-69','Age: 60-69','Age: 50-59'],
#     'Age_30Split': ['Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Less Than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30'],
#     'Age_40Split': ['Age Greater than 40','Age Less Than 40','Age Greater than 40','Age Less Than 40','Age Greater than 40','Age Less Than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40'],
#     'Age_50Split': ['Age Greater than 50','Age Less Than 50','Age Less Than 50','Age Less Than 50','Age Greater than 50','Age Less Than 50','Age Greater than 50','Age Greater than 50','Age Less Than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50'],
#     'Age_60Split': ['Age Less Than 60','Age Less Than 60','Age Less Than 60','Age Less Than 60','Age Greater than 60','Age Less Than 60','Age Greater than 60','Age Less Than 60','Age Less Than 60','Age Greater than 60','Age Greater than 60','Age Greater than 60','Age Greater than 60','Age Greater than 60','Age Less Than 60'],
#     'LC__PROFIT_COMPARE_3': ['Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is equally as profitable as other corporations','Belives that Walmart Is more profitable than other corporations','BLANK','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is equally as profitable as other corporations','BLANK','BLANK','BLANK','Belives that Walmart Is equally as profitable as other corporations']

# })

# chi_square_results = pd.DataFrame({
#     'IV': ['Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split'],
#     'Control_1': ['Age_50Split','Age_50Split','Age_40Split','Age_40Split','Age_50Split','Age_50Split','Age_60Split','AGE_RANGE'],
#     'Control_2': ['AGRMT_PHARMA_ROLE_1_COLLAPSED','GENDER','SAT_INFLUENCE_OF_BUS_3_COLLAPSED','LC__PROFIT_COMPARE_3','SAT_INFLUENCE_OF_BUS_3_COLLAPSED',"SAT_INFLUENCE_OF_BUS_3_COLLAPSED","COLLAR",None],
#     'IVLabel': ['Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30'],
#     'Control_1_Label': ["Age Greater than 50","Age Greater than 50","Age Less Than 40","Age Less Than 40","Age Less Than 50","Age Less Than 50","Age Less Than 60","Age: 70+"],
#     'Control_2_Label': ["Agreement that Pharmacists must distinguish between good and bad doctors: Strongly or somewhat disagree","Gender: Man",'Agreement with "Large corporations have positive effect on community": Strongly or somewhat agree',"Belives that Walmart Is equally as profitable as other corporations",'Agreement with "Large corporations have positive effect on community": Strongly or somewhat agree','Agreement with "Large corporations have positive effect on community": Strongly or somewhat disagree',"Collar: Blue Collar",None],
#     '%PL': [.08,.15,.0001,.0001,.0001,.71,.73,.0001],
#     'weighted_prediction': [.15,.22,.15,.15,.09,.51,.56,.09]
# })


# # Example filtered_results
# filtered_results = pd.DataFrame({
#     'IV': ['Age_30Split', 'Age_40Split', 'Age_50Split'],
#     'Control_1': ['Age_50Split', 'Age_50Split', 'Age_60Split'],
#     'Control_2': ['Age_40Split', np.nan, 'Age_30Split'],
#     'IVLabel': ['Age Greater than 30', 'Age Greater than 40', 'Age Less Than 50'],
#     'Control_1_Label': ['Age Less Than 50', 'Age Greater than 50', 'Age Less Than 60'],
#     'Control_2_Label': ['Age Greater than 40', np.nan, 'Age Greater than 30'],
#     'weighted_prediction': [0.12, 0.15, 0.08]
# })


import pandas as pd
from itertools import combinations
# import numpy as np
# from multiprocessing import Pool
from functions import create_weights, filter_results_by_combinations, match_jurors, generate_combinations




#Information about juror data file
data_file = "C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Data/FL_Opioids_JurorData.xlsx"
sheet_name = "use-labels"
dv = "DV_1PL_0Def" #DV variable
juror_id = "NAME" #Juror id variable

#Read in juror data file
juror_data_FL = pd.read_excel(data_file,sheet_name=sheet_name)

#Create a df of variables to use
use_cols_df = pd.read_excel(data_file, sheet_name="use cols")

#Create list of IVs for combinations
ivs = [var for var in use_cols_df['use_cols']]

#Identify variables to drop from
drop_cols = [col for col in juror_data_FL.columns if col not in ivs and col != dv and col != juror_id]

juror_data_FL = juror_data_FL.drop(columns = drop_cols)


#Read in chi-square results file
chi_square_results_FL = pd.read_excel('C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Data/FL_Opioids_DeDupedResults.xlsx')

#Create a df of plaintiff predictions from chi square results
plf_predictions_df = create_weights(chi_square_results_FL, .39, n_influence = 1.5)


#Create batches of combinations of 5 IVs
five_iv_combos = combinations(ivs, 5)

combination = next(five_iv_combos)
combinations = generate_combinations(combination)
filtered_results = filter_results_by_combinations(chi_square_results_FL, combinations)
matched_results = match_jurors(juror_data_FL, \
                               filtered_results,\
                               juror_id, \
                               'IV','IV_Label', \
                               'control_1','Control_1', \
                               'control_2','Control_2', \
                               'prediction')

for batch in five_iv_combos:
    combinations = generate_combinations(batch)
    filtered_results = filter_results_by_combinations(plf_predictions_df, combinations)
    matched_results = match_jurors(juror_data_FL, filtered_results)



print(matched_results)

for _,row in matched_results.iterrows():
    print(row['NAME'],row['weighted_prediction'])

for _,row in juror_data_FL.iterrows():
    if row['NAME'] in matched_results['NAME'].values:
        match_index = matched_results[matched_results['NAME'] == row['NAME']].index[0]
        # Pull a value from another column using that index
        prediction = matched_results.loc[match_index, 'weighted_prediction']
        outcome = row['DV_1PL_0Def']
        print(f"{row['NAME']}, predicted {prediction} %PL was {outcome}")
