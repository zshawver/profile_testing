
from itertools import combinations, islice, chain
from functions import generate_combinations, filter_results_by_combinations, match_jurors, preProcess_juror_data, preProcess_results_file, process_five_IV_tuple
import os
from multiprocessing import Pool, freeze_support, Manager
import time
import pandas as pd
import os
import logging
from datetime import datetime
import re
import math





def iter_combinations_in_chunks(ivs, chunk_size):
    """Yields chunks of 5-IV combinations as lists."""
    batch = []
    for combo in combinations(ivs, 5):
        batch.append(combo)
        if len(batch) == chunk_size:
            yield batch
            batch = []   # Reset batch
    if batch:
        yield batch








def process_chunk(batch, juror_data, juror_id, dv, chi_square_results, output_dir):
    """Worker function to process a batch of combinations and store results in a DataFrame."""
    process_id = os.getpid()

    # Create Logs directory
    log_dir = os.path.join(output_dir, "Logs")
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp
    date_time = datetime.now().isoformat()
    date_time = re.sub(r':[0-9]{2}\.[0-9]+', '', date_time)
    date_time = re.sub(r'T', '_', date_time)
    date_time = re.sub(r':', '-', date_time)

    # File paths for logs and CSVs (inside Logs folder)
    log_filename = os.path.join(log_dir, f"worker_{process_id}.log")
    output_filename = os.path.join(log_dir, f"{date_time}_worker_{process_id}.csv")
    summary_filename = os.path.join(log_dir, f"{date_time}_worker_summary_{process_id}.csv")

    results = []
    summary_results = []

    with open(log_filename, "a") as log_file:
        log_file.write(f"Worker {process_id} processing {len(batch)} combinations.\n")

        for combo in batch:
            combos = generate_combinations(combo)
            filtered_results = filter_results_by_combinations(chi_square_results, combos)
            matched_results = match_jurors(juror_data, filtered_results, juror_id, dv, 'prediction')

            log_file.write(f"{combo}: {filtered_results.shape[0]} chi-square results, matched {len(matched_results)} jurors\n")

            # Store summary data
            summary_results.append({
                "combination": combo,
                "num_filtered_results": filtered_results.shape[0],
                "num_jurors_matched": len(matched_results)
            })

            # Store detailed results
            for juror in matched_results:
                results.append({
                    "combination": combo,
                    "matched_juror": juror,
                    "num_filtered_results": filtered_results.shape[0],
                    "num_jurors_matched": len(matched_results)
                })

    # Save worker's detailed results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_filename, index=False)

    # Save worker's summary results
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(summary_filename, index=False)




def parallel_process(ivs, juror_data, juror_id, dv, chi_square_results, output_dir, num_workers=4, chunk_size=100):
    """Parallel execution to process chunks of combinations and store results."""
    print(f"Starting parallel processing with {num_workers} workers.")

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    chunk_gen = iter_combinations_in_chunks(ivs, chunk_size)

    with Pool(processes=num_workers) as pool:
        pool.starmap(process_chunk, [(batch, juror_data, juror_id, dv, chi_square_results, output_dir) for batch in chunk_gen])

    # Merge detailed and summary results
    merge_results(output_dir)
    merge_summary(output_dir)

def merge_results(output_dir):
    """Merge all worker CSVs into a single results file with timestamp."""
    log_dir = os.path.join(output_dir, "Logs")
    date_time = datetime.now().isoformat()
    date_time = re.sub(r':[0-9]{2}\.[0-9]+', '', date_time)
    date_time = re.sub(r'T', '_', date_time)
    date_time = re.sub(r':', '-', date_time)

    output_filename = os.path.join(output_dir, f"{date_time}_final_results.csv")
    all_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".csv") and "worker_" in f]

    if all_files:
        df_list = [pd.read_csv(f) for f in all_files]
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        print(f"Final results saved to {output_filename}")
    else:
        print("No data files found to merge.")

def merge_summary(output_dir):
    """Merge all worker summaries into a single summary file with timestamp."""
    log_dir = os.path.join(output_dir, "Logs")
    date_time = datetime.now().isoformat()
    date_time = re.sub(r':[0-9]{2}\.[0-9]+', '', date_time)
    date_time = re.sub(r'T', '_', date_time)
    date_time = re.sub(r':', '-', date_time)

    summary_filename = os.path.join(output_dir, f"{date_time}_combinations_summary.csv")
    all_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".csv") and "worker_summary_" in f]

    if all_files:
        df_list = [pd.read_csv(f) for f in all_files]
        summary_df = pd.concat(df_list, ignore_index=True)
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary saved to {summary_filename}")
    else:
        print("No summary data found to merge.")








# FINAL USABLE SCRIPT

if __name__ == "__main__":
    data_folder = "C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Data"
    output_dir = "C:/Users/zshawver/OneDrive - Dubin Consulting/Profile Testing/Output"
    juror_data_fn = "FL_Opioids_JurorData.xlsx"
    chisq_results_fn = "2025-01-29_11-02_FLOpioids_ProfileTesting_ResultsForProfileTesting.xlsx"
    juror_data_filepath = os.path.join(data_folder, juror_data_fn)
    chisq_results_filepath = os.path.join(data_folder, chisq_results_fn)
    data_sheet_name = "use-values"
    use_cols_sheet_name = "use cols-Full"
    dv = "DV_1PL_0Def"
    juror_id = "NAME"

    juror_data, ivs = preProcess_juror_data(juror_data_filepath, juror_id, dv, data_sheet_name, use_cols_sheet_name)
    chi_square_results = preProcess_results_file(chisq_results_filepath, ivs)

    freeze_support()
    parallel_process(ivs, juror_data, juror_id, dv, chi_square_results, output_dir, num_workers=16, chunk_size=10000)



# results_summary = pd.read_csv(os.path.join('Output',"2025-02-12_10-52_combinations_summary.csv"))

# results_df = pd.read_csv(os.path.join('Output',"2025-02-12_10-25_final_results.csv"))


batch_df = pd.read_excel("Output/MatchedJurors_Most_Results.xlsx",sheet_name="Sheet1")


# batch_df['median_prediction'] = batch_df.groupby('matched_juror')['prediction'].transform('median')

median_predictions = batch_df.groupby('matched_juror', as_index=False).agg(
    median_prediction=('prediction', 'median'),
    DV=('DV', 'first')
)

median_predictions['correctness'] = median_predictions.apply(
    lambda row: "correct" if (row['DV'] == 1 and row['median_prediction'] >= 0.5) or
                                (row['DV'] == 0 and row['median_prediction'] < 0.5)
                else "incorrect",
    axis=1
)

proportions = median_predictions['correctness'].value_counts(normalize=True)
correct_prop = proportions.get("correct", 0) * 100  # Convert to percentage
print(f"This batch correctly predicted {correct_prop:.1f}% of jurors")


correctness_by_dv = median_predictions.groupby("DV")["correctness"].value_counts(normalize=True).unstack()

# Extract proportions
correct_0 = correctness_by_dv.loc[0, "correct"] * 100 if "correct" in correctness_by_dv.columns else 0
correct_1 = correctness_by_dv.loc[1, "correct"] * 100 if "correct" in correctness_by_dv.columns else 0

# Print results
print(f"This batch correctly predicted {correct_1:.1f}% of Plaintiff jurors")
print(f"This batch correctly predicted {correct_0:.1f}% of Defense jurors")


num_predictions_below_0_5 = (median_predictions["median_prediction"] < 0.5).sum()

pct_predictions_below_0_5 = num_predictions_below_0_5 / median_predictions.shape[0]

print(f"{num_predictions_below_0_5} predictions below 0.5: {pct_predictions_below_0_5 * 100: .1f}% of all jurors")



# median_predictions.rename(columns={'prediction': 'median_prediction'}, inplace=True)
