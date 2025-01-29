import pymc as pm
import pandas as pd
import numpy as np
import arviz as az




# Step 1: Read and preprocess your dataset
# Replace with the path to your actual dataset
data_file = "C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Data/FL_Opioids_JurorData.xlsx"
sheet_name = "use-labels"
dv = "DV_1PL_0Def"

use_cols_df = pd.read_excel(data_file, sheet_name="use cols")

use_cols = [var for var in use_cols_df['use_cols']]

# Load data from Excel
df = pd.read_excel(data_file, sheet_name=sheet_name)

drop_cols = [col for col in df.columns if col not in use_cols and col != dv] #+ df.columns[df.isnull().sum() > 0].tolist()

df = df.drop(columns = drop_cols)

# Assume the last column is the binary target variable (y)
# and all other columns are predictors (X)
X = df.drop(columns = [dv]).values  # Predictor variables (features)
y = df[dv].values   # Binary outcome variable (target)

# Confirm data dimensions and types
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Step 2: Define the Bayesian logistic regression model
with pm.Model() as model:
    # Global shrinkage parameter (Horseshoe prior)
    tau = pm.HalfCauchy("tau", beta=1)

    # Local shrinkage parameters for each predictor
    lambda_ = pm.HalfCauchy("lambda_", beta=1, shape=X.shape[1])

    # Horseshoe prior for coefficients
    beta = pm.Normal("beta", mu=0, sigma=tau * lambda_, shape=X.shape[1])

    # Intercept term
    intercept = pm.Normal("intercept", mu=0, sigma=5)

    # Logistic regression logits
    logits = pm.math.dot(X, beta) + intercept
    p = pm.Deterministic("p", pm.math.sigmoid(logits))

    # Likelihood function
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    # Step 3: Fit the model using MCMC
    trace = pm.sample(
        draws=2000,  # Posterior draws
        #If precision in posterior inferences is poor (i.e., low MCSE)
        tune=1000,   # Tuning steps
        target_accept=0.9,  # Higher acceptance rate for better sampling
        return_inferencedata=True
    )

# Step 4: Summarize the posterior results
summary = az.summary(trace, hdi_prob=0.95)
print(summary)

# Save the results to a CSV for reporting
summary.to_csv("bayesian_logistic_regression_summary.csv")

# Step 5: (Optional) Visualize diagnostics and posterior distributions
az.plot_trace(trace)
az.plot_posterior(trace)
