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
def generate_combinations(batch, max_comb=3):
    for r in range(1, max_comb + 1):
        yield from combinations(batch, r)  # Yielding instead of storing


def filter_results_by_combinations(df, combinations):
    # Convert combinations to frozensets
    combo_sets = {frozenset(combo) for combo in combinations}

    # Vectorized filtering: Keep only rows with matching IV sets
    matching_rows = df[df['iv_sets'].apply(lambda x: x in combo_sets)]

    return matching_rows


# # Function to match juror responses with filtered results
# def match_jurors(juror_data, filtered_results,name,iv1,iv1_label,iv2,iv2_label,iv3,iv3label,prediction_column):
#     results = []

#     for _, row in filtered_results.iterrows():
#         # Extract relevant columns and labels from the current row
#         iv_col, c1_col, c2_col = row[iv1], row[iv2], row[iv3]
#         iv_label, c1_label, c2_label = row[iv1_label], row[iv2_label], row[iv3label]
#         prediction = row[prediction_column]

#         # Create boolean condition to match jurors
#         condition = (
#             (juror_data[iv_col] == iv_label) &
#             (juror_data[c1_col] == c1_label if pd.notna(c1_col) else True) &
#             (juror_data[c2_col] == c2_label if pd.notna(c2_col) else True)
#         )

#         # Filter matching jurors
#         matched_jurors = juror_data[condition]
#         for _, juror in matched_jurors.iterrows():
#             results.append({
#                 'NAME': juror[name],
#                 'weighted_prediction': prediction
#             })

#     return pd.DataFrame(results)

def match_jurors(juror_data, filtered_results, name_col, dv_col, batch_name, prediction_column):
    results = []

    # Step 1: Create frozensets of IV levels per juror
    def extract_iv_levels(row, iv_sets):
        return frozenset(row[iv] for iv in iv_sets if pd.notna(row[iv]))


    results_in_batch = filtered_results.shape[0]
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
                'batch': batch_name,
                'n_results_in_batch': results_in_batch,
                'iv_sets': row['iv_sets'],  # Store the frozen set directly for post-processing
                'iv_labels': row['iv_label_sets'],
                'prediction': prediction
            }
            for _, juror in matching_jurors.iterrows()
        ])

    return results  # Return a list of dictionaries for flexibility

def process_batch_of_IVs(batch,juror_data,juror_id, dv, plf_predictions_df):
    combos = generate_combinations(batch)
    filtered_results = filter_results_by_combinations(plf_predictions_df, combos)
    matched_results = match_jurors(juror_data, \
                                  filtered_results, \
                                  juror_id, \
                                  dv, \
                                  'test', \
                                  'prediction')
    return matched_results
