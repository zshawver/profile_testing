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


# Function to match juror responses with filtered results
def match_jurors(juror_data, filtered_results,name,iv1,iv1_label,iv2,iv2_label,iv3,iv3label,prediction_column):
    results = []

    for _, row in filtered_results.iterrows():
        # Extract relevant columns and labels from the current row
        iv_col, c1_col, c2_col = row[iv1], row[iv2], row[iv3]
        iv_label, c1_label, c2_label = row[iv1_label], row[iv2_label], row[iv3label]
        prediction = row[prediction_column]

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
