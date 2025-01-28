
import pandas as pd
from itertools import combinations
import numpy as np
from multiprocessing import Pool




# Generate all combinations for 1-IV, 2-IV, and 3-IV tests
def generate_combinations(batch, max_comb=3):
    all_combinations = []
    for r in range(1, max_comb + 1):
        all_combinations.extend(combinations(batch, r))
    return all_combinations

# Function to filter results by combinations
def filter_results_by_combinations(df, combinations):
    """
    Filters rows in `df` to only include those where the IVs match
    any of the provided combinations exactly.

    Args:
        df (pd.DataFrame): DataFrame containing chi-square results.
        combinations (list of lists): List of IV combinations to match.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only matching rows.
    """
    # Initialize a list to collect filtered DataFrames for each combo
    filtered_results = []

    for combo in combinations:
        combo_set = set(combo)  # Convert combo to a set

        # Filter rows where the IVs match the combo exactly
        matching_rows = df[
            df[['IV', 'Control_1', 'Control_2']].apply(
                lambda row: set(row.dropna()) == combo_set, axis=1
            )
        ]

        filtered_results.append(matching_rows)  # Add the matching rows

    # Concatenate all matching rows into a single DataFrame
    return pd.concat(filtered_results, ignore_index=True)

chi_square_results_FL = pd.read_excel('C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Data/FL_Opioids_DeDupedResults.xlsx')

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

# ivs = ['Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_30Split','Age_50Split','Age_50Split','Age_40Split','Age_40Split','Age_50Split','Age_50Split','Age_60Split','AGE_RANGE','AGRMT_PHARMA_ROLE_1_COLLAPSED','GENDER','SAT_INFLUENCE_OF_BUS_3_COLLAPSED','LC__PROFIT_COMPARE_3','SAT_INFLUENCE_OF_BUS_3_COLLAPSED',"SAT_INFLUENCE_OF_BUS_3_COLLAPSED","COLLAR",None]

ivs = list(set(iv for iv in chi_square_results_FL['IV']))

control1_ivs = list(set(iv for iv in chi_square_results_FL['control_1'] if iv not in ivs and iv is not np.nan))

control2_ivs = list(set(iv for iv in chi_square_results_FL['control_2'] if iv not in ivs and iv not in control1_ivs and iv is not np.nan))

allIVs = ivs+control1_ivs+control2_ivs

ivs_5_ways = combinations(allIVs, 5)

# ivs_5_ways_lst = list(map(list,ivs_5_ways))

#Get a single batch of 5 ivs combo
batch = next((item for item in reversed(ivs_5_ways) if 'Age_30Split' in item))

for batch in ivs_5_ways:
    combinations = generate_combinations(batch)

    filtered_results = filter_results_by_combinations(chi_square_results_FL, combinations)




# Example juror_data
juror_data = pd.DataFrame({
    'juror_name': ['John S','Bob R','Mike T','James W','Will F','Carl W','Tina T','Laura L','Candice B','Amy P','Rachel D','Seth M','Adam S','Jason B','Kristen W'],
    'DV_1PL_0Def': ['Defense','Plaintiff','Plaintiff','Defense','Defense','Plaintiff','Defense','Plaintiff','Defense','Plaintiff','Defense','Defense','Plaintiff','Defense','Plaintiff'],
    'GENDER': ['Gender: Man','Gender: Woman','Gender: Man','Gender: Woman','Gender: Man','Gender: Man','Gender: Man','Gender: Man','Gender: Woman','Gender: Man','Gender: Woman','Gender: Woman','Gender: Woman','Gender: Man','Gender: Woman'],
    'RACE': ['Race: White / Caucasian','Race: White / Caucasian','Race: White / Caucasian','Race: Black / African American','Race: White / Caucasian','Race: Hispanic / Latinx','Race: White / Caucasian','Race: Hispanic / Latinx','Race: Black / African American','Race: Black / African American','Race: White / Caucasian','Race: White / Caucasian','Race: Black / African American','Race: Black / African American','Race: Black / African American'],
    'PA': ['Political Affiliation: Democrat','Political Affiliation: Not Registered / Unaffiliated / Other','Political Affiliation: Democrat','Political Affiliation: Not Registered / Unaffiliated / Other','Political Affiliation: Democrat','Political Affiliation: Independent','Political Affiliation: Democrat','Political Affiliation: Republican','Political Affiliation: Democrat','Political Affiliation: Democrat','Political Affiliation: Republican','Political Affiliation: Democrat','Political Affiliation: Democrat','Political Affiliation: Independent','Political Affiliation: Democrat'],
    'AGE_RANGE': ['Age: 50-59','Age: 30-39','Age: 40-49','Age: 20-29','Age: 60-69','Age: 30-39','Age: 60-69','Age: 50-59','Age: 40-49','Age: 60-69','Age: 60-69','Age: 70+','Age: 60-69','Age: 60-69','Age: 50-59'],
    'Age_30Split': ['Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Less Than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30','Age Greater than 30'],
    'Age_40Split': ['Age Greater than 40','Age Less Than 40','Age Greater than 40','Age Less Than 40','Age Greater than 40','Age Less Than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40','Age Greater than 40'],
    'Age_50Split': ['Age Greater than 50','Age Less Than 50','Age Less Than 50','Age Less Than 50','Age Greater than 50','Age Less Than 50','Age Greater than 50','Age Greater than 50','Age Less Than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50','Age Greater than 50'],
    'Age_60Split': ['Age Less Than 60','Age Less Than 60','Age Less Than 60','Age Less Than 60','Age Greater than 60','Age Less Than 60','Age Greater than 60','Age Less Than 60','Age Less Than 60','Age Greater than 60','Age Greater than 60','Age Greater than 60','Age Greater than 60','Age Greater than 60','Age Less Than 60'],
    'LC__PROFIT_COMPARE_3': ['Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is equally as profitable as other corporations','Belives that Walmart Is more profitable than other corporations','BLANK','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is more profitable than other corporations','Belives that Walmart Is equally as profitable as other corporations','BLANK','BLANK','BLANK','Belives that Walmart Is equally as profitable as other corporations']

})

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

# Function to match juror responses with filtered results
def match_jurors(juror_data, filtered_results):
    results = []

    for _, row in filtered_results.iterrows():
        # Extract relevant columns and labels from the current row
        iv_col, c1_col, c2_col = row['IV'], row['Control_1'], row['Control_2']
        iv_label, c1_label, c2_label = row['IVLabel'], row['Control_1_Label'], row['Control_2_Label']
        prediction = row['weighted_prediction']

        # Create boolean condition to match jurors
        condition = (
            (juror_data[iv_col] == iv_label) &
            (juror_data[c1_col] == c1_label if pd.notna(c1_col) else True) &
            (juror_data[c2_col] == c2_label if pd.notna(c2_col) else True)
        )

        # Filter matching jurors
        matched_jurors = juror_data[condition]
        for _, juror in matched_jurors.iterrows():
            results.append({
                'NAME': juror['NAME'],
                'weighted_prediction': prediction
            })

    return pd.DataFrame(results)

juror_data_FL = pd.read_excel('C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Data/FL_Opioids_JurorData.xlsx',sheet_name='use-labels')

# Apply function
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
