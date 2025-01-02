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
    p_plaintiff = row["n_plaintiff"] / total
    p_defense = row["n_defense"] / total
    if p_plaintiff == 0 or p_defense == 0:  # Avoid log(0)
        return 0
    entropy = -(
        p_plaintiff * np.log(p_plaintiff) + p_defense * np.log(p_defense)
    )
    return entropy

def create_weights(chisq_df, pro_plaintiff_threshold, pro_defense_threshold):
    # Categorize results into sides
    chisq_df["side"] = np.where(
        chisq_df["percent_plaintiff"] > pro_plaintiff_threshold, "Pro-Plaintiff",
        np.where(chisq_df["percent_plaintiff"] < pro_defense_threshold, "Pro-Defense", "Neutral")
    )

    # Calculate total counts for each side
    side_counts = chisq_df["side"].value_counts()
    total_pro_plaintiff = side_counts.get("Pro-Plaintiff", 0)
    total_pro_defense = side_counts.get("Pro-Defense", 0)

    # Add normalization factor for Pro-Plaintiff and Pro-Defense
    chisq_df["side_weight"] = chisq_df["side"].map({
        "Pro-Plaintiff": 1 / total_pro_plaintiff if total_pro_plaintiff > 0 else 0,
        "Pro-Defense": 1 / total_pro_defense if total_pro_defense > 0 else 0,
        "Neutral": 0  # Neutral cases are not weighted
    })

    # 1. Logarithmic Weighting
    chisq_df["log_weight"] = np.log(chisq_df["N"] + 1)

    # 2. Split Difference Scaling
    chisq_df["split_difference"] = abs(chisq_df["n_plaintiff"] - chisq_df["n_defense"])
    chisq_df["split_weight"] = chisq_df["split_difference"] / chisq_df["N"]

    # 3. Bayesian Shrinkage
    # Assuming an overall baseline decision rate
    overall_decision_rate = 0.6
    chisq_df["bayesian_weight"] = (
        chisq_df["N"] * chisq_df["percent_plaintiff"] + overall_decision_rate * 100
    ) / (chisq_df["N"] + 1)

    # 4. Quadratic Weighting
    chisq_df["quadratic_weight"] = np.sqrt(chisq_df["N"])

    # 5. Entropy-Based Weighting
    chisq_df["entropy"] = chisq_df.apply(calculate_entropy, axis=1)
    chisq_df["entropy_weight"] = 1 - chisq_df["entropy"]

    # 6. Hybrid Approach
    chisq_df["hybrid_weight"] = (
        chisq_df["log_weight"]
        * chisq_df["split_weight"]
        * chisq_df["entropy_weight"]
    )

    # Apply side-based normalization to all weights
    weighting_schemes = [
        "log_weight", "split_weight", "bayesian_weight",
        "quadratic_weight", "entropy_weight", "hybrid_weight"
    ]

    for scheme in weighting_schemes:
        chisq_df[f"adjusted_{scheme}"] = chisq_df[scheme] * chisq_df["side_weight"]

    # Optional: Normalize adjusted weights to [0, 1] for comparison
    for scheme in weighting_schemes:
        adjusted_column = f"adjusted_{scheme}"
        chisq_df[f"normalized_{adjusted_column}"] = (
            chisq_df[adjusted_column] - chisq_df[adjusted_column].min()
        ) / (chisq_df[adjusted_column].max() - chisq_df[adjusted_column].min())

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
