import numpy as np
import pandas as pd
from functions import calculate_entropy

# Assuming chisq_df is already loaded with columns:
# - "N": Total number of people in the test.
# - "percent_plaintiff": Percentage of jurors deciding for Plaintiff.
# - "n_plaintiff": Number of jurors deciding for Plaintiff.
# - "n_defense": Number of jurors deciding for Defense.

# Define thresholds for Pro-Plaintiff and Pro-Defense
pro_plaintiff_threshold = .7  # Percentage threshold for Pro-Plaintiff
pro_defense_threshold = .2    # Percentage threshold for Pro-Defense

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

# Inspect the results
print(chisq_df.head())
