import numpy as np
import pandas as pd
from juror import Juror

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

def create_weights(chisq_df, baseline_plf):
    # Apply a minimum weight for zero pct_1 values
    # chisq_df["adjusted_pct_1"] = chisq_df["pct_1"].apply(lambda x: max(x, 0.001))  # 0.001 to prevent zero issues
    # chisq_df["adjusted_pct_1"] = chisq_df["pct_1"].apply(lambda x: min(x, .999))  # 0.001 to prevent zero issues

    # Create a weight by N
    # chisq_df["weight_by_N"] = chisq_df["N"] / chisq_df["N"].max()

    # # Logarithmic Weighting
    # chisq_df["log_weight_N"] = np.log(chisq_df["N"] + 1)

    # # Quadratic Weighting
    # chisq_df["quadratic_weight_N"] = (chisq_df["N"]**2)/(chisq_df["N"].max()**2)

    # # Split Difference Weighting
    # chisq_df["split_difference"] = abs(chisq_df["dv_1"] - chisq_df["dv_0"])
    # chisq_df["split_weight_log"] = chisq_df["split_difference"] / np.log(chisq_df["N"] + 1)
    # chisq_df["split_weight_quadratic"] = chisq_df["split_difference"] / np.sqrt(chisq_df["N"])

    # Bayesian Shrinkage
    # chisq_df["bayesian_weight"] = (
    #     chisq_df["N"] * chisq_df["adjusted_pct_1"] + baseline_plf
    # ) / (chisq_df["N"] + 1)

    # Entropy-Based Weighting
    # chisq_df["entropy"] = chisq_df.apply(calculate_entropy, axis=1)
    # chisq_df["entropy_weight"] = 1 - chisq_df["entropy"]

    # # Hybrid Approach
    # chisq_df["hybrid_weight"] = (
    #     chisq_df["log_weight_N"] * chisq_df["split_weight_log"] * chisq_df["entropy_weight"]
    # )

    # # Normalization of weights
    # weight_columns = [
    #     "log_weight_N", "quadratic_weight_N", "split_weight_log",
    #     "split_weight_quadratic", "bayesian_weight", "entropy_weight", "hybrid_weight"
    # ]

    # Hybrid Approach
    # chisq_df["hybrid_weight"] = (
    #     chisq_df["weight_by_N"] * chisq_df["bayesian_weight"] * chisq_df["entropy_weight"]
    # )

    # Normalization of weights
    # weight_columns = [
    #     "weight_by_N", "bayesian_weight", "entropy_weight", "hybrid_weight"
    # ]

    # weight_columns = [
    #     "bayesian_weight"
    # ]

    # for col in weight_columns:
    #     normalized_col = f"normalized_{col}"
    #     chisq_df[normalized_col] = (chisq_df[col] - chisq_df[col].min()) / (chisq_df[col].max() - chisq_df[col].min())

    # # Apply normalized weights to Plaintiff Percentage for predictions
    # for col in weight_columns:
    #     normalized_col = f"normalized_{col}"  # Use the normalized version of the weight
    #     prediction_col = f"{col}_prediction"
    #     chisq_df[prediction_col] = chisq_df[normalized_col] * chisq_df["adjusted_pct_1"]

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
    chisq_df["weight"] = (chisq_df["scaled_lnN"]*1.25)*chisq_df["normalized_dev"]
    #Normalize the weight variable
    chisq_df["normalized_weight"] = (chisq_df["weight"] - chisq_df["weight"].min())/(chisq_df["weight"].max()-chisq_df["weight"].min())

    #Creating predictions for %Plaintiff

    #Based on weight
    chisq_df["prediction_weight"] = ((1-chisq_df["weight"])*baseline_plf+(chisq_df["weight"]*chisq_df["pct_1"]))
    #Based on normalized weight
    chisq_df["prediction_normalized_weight"] = ((1-chisq_df["normalized_weight"])*baseline_plf+(chisq_df["normalized_weight"]*chisq_df["pct_1"]))




    drop_columns = [col for col in chisq_df if "_prediction" not in col and col != "iv_level_tuples" and col != "pct_1" and col != "N"]

    chisq_df = chisq_df.drop(columns = drop_columns)

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
