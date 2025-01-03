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

def create_weights(chisq_df):
    # 1. Logarithmic Weighting
    chisq_df["log_weight"] = 1 - (1 / np.log(chisq_df["N"] + 1))

    # 2. Quadratic Weighting
    chisq_df["quadratic_weight"] = 1 - (1 / np.sqrt(chisq_df["N"]))

    # 3a. Split Difference Weighting
    chisq_df["split_difference"] = abs(chisq_df["dv_1"] - chisq_df["dv_0"])

    # 3b. Split Difference Weighting - Log Scaling
    chisq_df["split_weight_log"] = chisq_df["split_difference"] / chisq_df["N"]

    # 3c. Split Difference Weighting - Quadratic Scaling
    chisq_df["split_weight_quadratic"] = chisq_df["split_difference"] / np.sqrt(chisq_df["N"])

    # 4. Bayesian Shrinkage
    overall_decision_rate = 0.6
    chisq_df["bayesian_weight"] = (
        chisq_df["N"] * chisq_df["pct_1"] + overall_decision_rate * 100
    ) / (chisq_df["N"] + 1)

    # 5. Entropy-Based Weighting
    chisq_df["entropy"] = chisq_df.apply(calculate_entropy, axis=1)
    chisq_df["entropy_weight"] = 1 - chisq_df["entropy"]

    # 6. Hybrid Approach
    chisq_df["hybrid_weight"] = (
        chisq_df["log_weight"]
        * chisq_df["split_weight_log"]
        * chisq_df["entropy_weight"]
    )


    # Normalize adjusted weights to [0, 1] for comparison
    weighting_schemes = [
        "log_weight_adjusted", "quadratic_weight_adjusted", "split_weight_log_adjusted",
        "split_weight_quadratic_adjusted", "bayesian_weight_adjusted", "entropy_weight_adjusted", "hybrid_weight_adjusted"
    ]

    for scheme in weighting_schemes:
        adjusted_column = f"normalized_{scheme}"
        chisq_df[adjusted_column] = (
            chisq_df[scheme] - chisq_df[scheme].min()
        ) / (chisq_df[scheme].max() - chisq_df[scheme].min())

    # Apply weights to Plaintiff Percentage to make predictions
    for scheme in weighting_schemes:
        prediction_column = f"{scheme}_prediction"
        chisq_df[prediction_column] = chisq_df[scheme] * chisq_df["pct_1"]

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
