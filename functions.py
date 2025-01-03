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
    # Logarithmic Weighting
    chisq_df["log_weight_N"] = 1 - (1 / np.log(chisq_df["N"] + 1))

    # Quadratic Weighting
    chisq_df["quadratic_weight_N"] = 1 - (1 / np.sqrt(chisq_df["N"]))

    # Split Difference Weighting
    chisq_df["split_difference"] = abs(chisq_df["dv_1"] - chisq_df["dv_0"])
    chisq_df["split_weight_log"] = chisq_df["split_difference"] / chisq_df["N"]
    chisq_df["split_weight_quadratic"] = chisq_df["split_difference"] / np.sqrt(chisq_df["N"])

    # Bayesian Shrinkage
    chisq_df["bayesian_weight"] = (
        chisq_df["N"] * chisq_df["pct_1"] + baseline_plf
    ) / (chisq_df["N"] + 1)

    # Entropy-Based Weighting
    chisq_df["entropy"] = chisq_df.apply(calculate_entropy, axis=1)
    chisq_df["entropy_weight"] = 1 - chisq_df["entropy"]

    # Hybrid Approach
    chisq_df["hybrid_weight"] = (
        chisq_df["log_weight_N"] * chisq_df["split_weight_log"] * chisq_df["entropy_weight"]
    )

    # Normalization of weights
    weight_columns = [
        "log_weight_N", "quadratic_weight_N", "split_weight_log",
        "split_weight_quadratic", "bayesian_weight", "entropy_weight", "hybrid_weight"
    ]

    for col in weight_columns:
        normalized_col = f"normalized_{col}"
        chisq_df[normalized_col] = (chisq_df[col] - chisq_df[col].min()) / (chisq_df[col].max() - chisq_df[col].min())

    # Apply normalized weights to Plaintiff Percentage for predictions
    for col in weight_columns:
        prediction_col = f"{col}_prediction"
        chisq_df[prediction_col] = chisq_df[col] * chisq_df["pct_1"]

    drop_columns = [col for col in chisq_df if "_prediction" not in col and col != "iv_level_tuples" and col != "pct_1"]

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
