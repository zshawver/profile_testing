import numpy as np
import pandas as pd
from juror import Juror
from itertools import combinations

def prepare_juror_lists(df: pd.DataFrame, dv, juror_name: str, plaintiff_label, defense_label) -> dict:

    #Empty dictionary to hold jurors' information
    jurors = {}

    #Add jurors to dictionary, jurorNumber:Juror
    for i, row in df.iterrows():
        jurorNumber = "j{}".format(i+1)
        name = row[juror_name] #get juror's name
        ivDict = row.drop(columns = juror_name).to_dict() #Rest of juror's responses
        jurors[jurorNumber] = Juror(name,ivDict)

    #Separate out plaintiff and defense jurors
    return jurors, {jNum:j for jNum,j in jurors.items() if getattr(j,dv) == plaintiff_label}, {jNum:j for jNum,j in jurors.items() if getattr(j,dv) == defense_label}

def calculate_entropy(row):
    total = row["N"]
    if total == 0:
        return 0
    p_plaintiff = row["dv_1"] / total
    p_defense = row["dv_0"] / total
    if p_plaintiff == 0 or p_defense == 0:  # Avoid log(0)
        return 0
    entropy = -(
        p_plaintiff * np.log(p_plaintiff) + p_defense * np.log(p_defense)
    )
    return entropy

def create_weights(chisq_df, baseline_plf, n_influence = 1, dev_influence = 1):
    #Normalize deviations in each direction

    #Get deviation from baseline for each plaintiff result
    chisq_df["dev"] = chisq_df["pct_1"] - baseline_plf
    #Get min and max deviations for both pro-PL and pro-DF deviations
    min_PL_Dev = chisq_df.loc[chisq_df["dev"] >= 0, "dev"].min()
    max_PL_Dev = chisq_df.loc[chisq_df["dev"] >= 0, "dev"].max()
    min_DF_Dev = chisq_df.loc[chisq_df["dev"] < 0, "dev"].min()
    max_DF_Dev = chisq_df.loc[chisq_df["dev"] < 0, "dev"].max()
    chisq_df["normalized_dev"] = np.where(
        chisq_df["dev"] >= 0, (chisq_df["dev"]-min_PL_Dev)/(max_PL_Dev-min_PL_Dev),
        (chisq_df["dev"]-max_DF_Dev)/(min_DF_Dev-max_DF_Dev)
        )

    #Scale N by the max N for each test type

    #Get max N for each test type
    max_N_ME = chisq_df.loc[chisq_df["type"] == "ME", "N"].max()
    max_N_2Way = chisq_df.loc[chisq_df["type"] == "2-Way", "N"].max()
    max_N_3Way = chisq_df.loc[chisq_df["type"] == "3-Way", "N"].max()
    #Get the natural log of N
    chisq_df["lnN"] = np.log(chisq_df["N"])
    #Divide each logged N by the natural log of N for its test type
    chisq_df["scaled_lnN"] = np.where(
        chisq_df["type"] == "ME", chisq_df["lnN"]/np.log(max_N_ME),
        np.where(
            chisq_df["type"] == "2-Way", chisq_df["lnN"]/np.log(max_N_2Way),
            chisq_df["lnN"]/np.log(max_N_3Way)
            )
        )

    #Create a weight variable that gives additional influence to sample size
    chisq_df["weight"] = (chisq_df["scaled_lnN"]*n_influence)*(chisq_df["normalized_dev"]*dev_influence)
    # #Normalize the weight variable
    # chisq_df["normalized_weight"] = (chisq_df["weight"] - chisq_df["weight"].min())/(chisq_df["weight"].max()-chisq_df["weight"].min())

    #Creating predictions for %Plaintiff

    #Based on weight
    chisq_df["prediction"] = ((1-chisq_df["weight"])*baseline_plf+(chisq_df["weight"]*chisq_df["pct_1"]))
    # #Based on normalized weight
    # chisq_df["prediction_normalized_weight"] = ((1-chisq_df["normalized_weight"])*baseline_plf+(chisq_df["normalized_weight"]*chisq_df["pct_1"]))


    drop_cols = ["pct_0","type","dev","normalized_dev","lnN","scaled_lnN","weight"]

    chisq_df = chisq_df.drop(columns = drop_cols)

    return chisq_df


# Function to create a tuple excluding NaN values
def create_nested_tuples(row):
    nested_tuples = []
    for iv, level in zip(
        [row['control_2'], row['control_1'], row['iv']],
        [row['control_2_level'], row['control_1_level'], row['iv_level']]
    ):
        if pd.notna(iv) and pd.notna(level):  # Include only non-NaN pairs
            nested_tuples.append((iv, level))
    return tuple(nested_tuples)


# Function to check if all non-NaN IVs in the row are in the list
def row_in_ivs(row, ivs):
    row = row.dropna()  # Remove NaN values
    return all(iv in ivs for iv in row)

# Generate all combinations for 1-IV, 2-IV, and 3-IV tests
def generate_combinations(five_iv_tuple: tuple, max_comb=3):
    for r in range(1, max_comb + 1):
        yield from combinations(five_iv_tuple, r)  # Yielding instead of storing


def filter_results_by_combinations(df, combos):
    # Convert combinations to frozensets
    combo_sets = {frozenset(combo) for combo in combos}

    # Vectorized filtering: Keep only rows with matching IV sets
    matching_rows = df[df['iv_sets'].apply(lambda x: x in combo_sets)]

    return matching_rows

def match_jurors(juror_data, filtered_results, name_col, dv_col, prediction_column):
    results = []

    # Step 1: Create frozensets of IV levels per juror
    def extract_iv_levels(row, iv_sets):
        return frozenset(row[iv] for iv in iv_sets if pd.notna(row[iv]))


    results_in_five_iv_tuple = filtered_results.shape[0]
    # Step 2: Loop over filtered results
    for _, row in filtered_results.iterrows():

        results_iv_set = row['iv_sets']

        results_iv_level_set = row['iv_level_sets']

        juror_data['juror_iv_level_sets'] = juror_data.apply(lambda juror_row: extract_iv_levels(juror_row, results_iv_set), axis=1)

        prediction = row[prediction_column]

        # Step 3: Efficient filtering using .apply()
        matching_jurors = juror_data[juror_data['juror_iv_level_sets'].apply(lambda x: x == results_iv_level_set)]

        # Step 4: Store results efficiently
        results.extend([
            {
                'juror_name': juror[name_col],
                'dv': juror[dv_col],
                'n_results': results_in_five_iv_tuple,
                'iv_sets': row['iv_sets'],  # Store the frozen set directly for post-processing
                'iv_labels': row['iv_label_sets'],
                'prediction': prediction
            }
            for _, juror in matching_jurors.iterrows()
        ])

    return results  # Return a list of dictionaries for flexibility

def process_five_IV_tuple(five_iv_tuple: tuple,juror_data,juror_id, dv, chi_square_results):
    combos = generate_combinations(five_iv_tuple)
    filtered_results = filter_results_by_combinations(chi_square_results, combos)
    matched_results = match_jurors(juror_data, \
                                  filtered_results, \
                                  juror_id, \
                                  dv, \
                                  'prediction')
    return matched_results

def preProcess_juror_data(juror_data_filepath,juror_id,dv,data_sheet_name,use_cols_sheet_name):

    #Read in juror data file
    juror_data = pd.read_excel(juror_data_filepath,sheet_name=data_sheet_name)

    #Create a df of variables to use
    use_cols_df = pd.read_excel(juror_data_filepath, sheet_name=use_cols_sheet_name)

    #Create list of IVs for combinations
    ivs = [var for var in use_cols_df['use_cols']]

    #Identify variables to drop from juror dataset
    drop_cols = [col for col in juror_data.columns if col not in ivs and col != dv and col != juror_id]

    #Drop unused columns from juror data sheet
    juror_data = juror_data.drop(columns = drop_cols)

    return juror_data, ivs

def preProcess_results_file(chisq_results_filepath,ivs):


    #Read in chi-square results file
    chi_square_results = pd.read_excel(chisq_results_filepath)

    #Filter out results with an not in ivs list
    chi_square_results = chi_square_results[chi_square_results[['iv1', 'iv2', 'iv3']].apply(lambda row: row_in_ivs(row, ivs), axis=1)]


    #Create a df of plaintiff predictions from chi square results
    chi_square_results = create_weights(chi_square_results, .39, n_influence = 1.5)

    #Create columns that combine IV names, IV levels (i.e., values) and IV labels
    chi_square_results["iv_sets"] = chi_square_results[['iv1', 'iv2', 'iv3']].apply(lambda row: frozenset(row.dropna()), axis=1)
    chi_square_results["iv_level_sets"] = chi_square_results[['iv1_level', 'iv2_level', 'iv3_level']].apply(lambda row: frozenset(row.dropna()), axis=1)
    chi_square_results["iv_label_sets"] = chi_square_results[['iv1_label', 'iv2_label', 'iv3_label']].apply(lambda row: frozenset(row.dropna()), axis=1)

    return chi_square_results
