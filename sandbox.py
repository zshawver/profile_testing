

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
import os

# Function to check if all non-NaN IVs in the row are in the list
def row_in_ivs(row, ivs):
    row = row.dropna()  # Remove NaN values
    return all(iv in ivs for iv in row)

#Information about juror data file
data_folder = "C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Data"
juror_data_fn = "FL_Opioids_JurorData.xlsx"
chisq_results_fn = "2025-01-29_11-02_FLOpioids_ProfileTesting_ResultsForProfileTesting.xlsx"
juror_data_filepath = os.path.join(data_folder, juror_data_fn)
chisq_results_filepath = os.path.join(data_folder, chisq_results_fn)
sheet_name = "use-values"
dv = "DV_1PL_0Def" #DV variable
juror_id = "NAME" #Juror id variable

#Read in juror data file
juror_data_FL = pd.read_excel(juror_data_filepath,sheet_name=sheet_name)

#Create a df of variables to use
use_cols_df = pd.read_excel(juror_data_filepath, sheet_name="use cols")

#Create list of IVs for combinations
ivs = [var for var in use_cols_df['use_cols']]

#Identify variables to drop from juror dataset
drop_cols = [col for col in juror_data_FL.columns if col not in ivs and col != dv and col != juror_id]

juror_data_FL = juror_data_FL.drop(columns = drop_cols)






#Read in chi-square results file
chi_square_results_FL = pd.read_excel(chisq_results_filepath)

#Filter out results with an not in ivs list
chi_square_results_FL = chi_square_results_FL[chi_square_results_FL[['iv1', 'iv2', 'iv3']].apply(lambda row: row_in_ivs(row, ivs), axis=1)]


#Create a df of plaintiff predictions from chi square results
plf_predictions_df = create_weights(chi_square_results_FL, .39, n_influence = 1.5)
# Precompute unique row IV sets
plf_predictions_df["iv_sets"] = plf_predictions_df[['iv1', 'iv2', 'iv3']].apply(lambda row: frozenset(row.dropna()), axis=1)


#Create batches of combinations of 5 IVs
five_iv_combos = combinations(ivs, 5)

# combination = next(five_iv_combos)
# combinations = generate_combinations(combination)
# filtered_results = filter_results_by_combinations(plf_predictions_df, combinations)
# matched_results = match_jurors(juror_data_FL, \
#                                filtered_results,\
#                                juror_id, \
#                                'IV','IV_Label', \
#                                'control_1','Control_1', \
#                                'control_2','Control_2', \
#                                'prediction')

# for batch in five_iv_combos:
#     combinations = generate_combinations(batch)
#     filtered_results = filter_results_by_combinations(plf_predictions_df, combinations)
#     matched_results = match_jurors(juror_data_FL, filtered_results)



# print(matched_results)

# for _,row in matched_results.iterrows():
#     print(row['NAME'],row['weighted_prediction'])

# for _,row in juror_data_FL.iterrows():
#     if row['NAME'] in matched_results['NAME'].values:
#         match_index = matched_results[matched_results['NAME'] == row['NAME']].index[0]
#         # Pull a value from another column using that index
#         prediction = matched_results.loc[match_index, 'weighted_prediction']
#         outcome = row['DV_1PL_0Def']
#         print(f"{row['NAME']}, predicted {prediction:.0%} PL was {outcome}")


ivs_small = ivs[:7]

small_iv_combos = combinations(ivs_small, 5)
batch = next(small_iv_combos)



# start_time = time.time()

# all_results = []
# for i, combination in enumerate(small_iv_combos):
#     combinations = generate_combinations(combination)
#     filtered_results = filter_results_by_combinations(plf_predictions_df, combinations)
#     matched_results = match_jurors(juror_data_FL, filtered_results, juror_id,
#                                    'IV', 'IV_Label', 'control_1', 'Control_1',
#                                    'control_2', 'Control_2', 'prediction')
#     all_results.append(matched_results)
#     print(i)

# # Save output
# df_final = pd.concat(all_results)
# # df_final.to_parquet("juror_predictions.parquet", index=False)

# end_time = time.time()
# print(f"Serial Execution Time: {end_time - start_time:.2f} seconds")


def filter_results_by_combinations_new(df, combinations):
    # Convert combinations to frozensets
    combo_sets = {frozenset(combo) for combo in combinations}

    # Vectorized filtering: Keep only rows with matching IV sets
    # matching_rows = df[df['iv_sets'].isin(combo_sets)]
    matching_rows = df[df['iv_sets'].apply(lambda x: x in combo_sets)]

    return matching_rows



# start_time = time.time()
import timeit
def combo_and_filter(batch,plf_predictions_df):
    combos = generate_combinations(batch)
    filtered_results = filter_results_by_combinations(plf_predictions_df, combos)

def combo_and_filter_new(batch,plf_predictions_df):
    combos = generate_combinations(batch)
    filtered_results = filter_results_by_combinations_new(plf_predictions_df, combos)
    return filtered_results


batch = ('Age_30Split', 'SAT_INFLUENCE_OF_BUS_1_COLLAPSED', 'LC__PROFIT_COMPARE_3', 'COLLAR', 'ASSOCIATION')
combos = generate_combinations(batch)
filtered_results = filter_results_by_combinations_new(plf_predictions_df, combos)
# filtered_results = combo_and_filter_new(batch,plf_predictions_df)

other_filtered_results = combo_and_filter_new(batch,plf_predictions_df)


# for combo in combos:
#     print(combo)

execution_time = timeit.timeit('combo_and_filter_new(batch,plf_predictions_df)', globals=globals(), number=2000)

print(f"Average Execution Time: {execution_time / 2000:.10f} seconds")


# end_time = time.time()
# total_time_list = end_time-start_time
# total_time_generator = end_time-start_time
# print(f"Execution Time: {end_time - start_time:.1000f} seconds")
