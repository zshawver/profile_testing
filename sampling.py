import pandas as pd
import os
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from juror import Juror
import time
import re
from datetime import datetime

def prepare_juror_lists(df: pd.DataFrame,dv: str,juror_name: str,plaintiff_label: str,defense_label: str) -> dict:

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

def prepare_random_samples_of_jurors(N: int, iterations: int, pJurors: dict, dJurors: dict) -> list:
    #Get juror numbers for plaintiff and defense
    plaintiffJNums = np.array(list(plaintiffJurors.keys()))
    defenseJNums = np.array(list(defenseJurors.keys()))

    #Get Jurors for plaintiff and defense
    plaintiffJurorList = np.array(list(plaintiffJurors.values()))
    defenseJurorList = np.array(list(defenseJurors.values()))
    #Get a random sample of indices representing jurors
    plaintiffSampleIDXs = [np.random.choice(len(plaintiffJNums), size=N, replace=False) for i in range(iterations)]
    defenseSampleIDXs = [np.random.choice(len(defenseJNums), size=N, replace=False) for i in range(iterations)]

    #Gather the random sample of jurors into a list of lists
    plaintiffSamples = [list(plaintiffJurorList[idx]) for idx in plaintiffSampleIDXs]
    defenseSamples = [list(defenseJurorList[idx]) for idx in defenseSampleIDXs]

    #Combine the plaintiff and defense samples into a single list of lists
    jurorSamples = plaintiffSamples+defenseSamples
    return jurorSamples, plaintiffSamples, defenseSamples

def prepare_iv_lists(df: pd.DataFrame) -> list:

    #IV lists
    ivs = [col for col in df] #Individual IVs
    ivs_2ways = list(combinations(ivs, 2)) #2-IV combos
    ivs_3ways = list(combinations(ivs, 3)) #3-IV combos
    ivs_4ways = list(combinations(ivs, 4)) #4-IV combos

    #Gather all combinations of IVs into a single list
    return ivs_2ways+ivs_3ways+ivs_4ways

def prepare_sample_ivs_list(samples: list, iv_combos: list) -> list:
    return [[sample,iv_combo] for iv_combo in iv_combos for sample in samples]

def count_jurors_by_iv_combination(sample, iv_combination):
    #Create a counter to store counts for this sample
    iv_level_counts = Counter()

    #Do for all jurors
    for juror in sample:
        #Get each level for each IV in combination
        jurorLevels = [getattr(juror,iv) for iv in iv_combination]
        #Make sure that none of the levels are "BLANK"
        if "BLANK" not in jurorLevels:
            #Convert the levels to a tuple
            count_key = tuple(jurorLevels)
            #Add the tuple to the counter
            iv_level_counts[count_key] += 1

    return iv_level_counts

def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_chunk(chunk):
    return [count_jurors_by_iv_combination(s_iv[0], s_iv[1]) for s_iv in chunk]


def count_iv_combos_parallel(samples_ivs_list, chunk_size):
    chunks = list(chunkify(samples_ivs_list, chunk_size))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        sampleCounts = [result for future in futures for result in future.result()]

    return sampleCounts

def combine_counters(counter_list):
    combined = Counter()
    for count in counter_list:
        combined += count
    return combined


if __name__ == "__main__":
    #Key values in dataset
    folder: str = "C:\\Users\\zshawver\\OneDrive - Dubin Consulting\\Deselection Profile Analyses\\State Farm GA\\Data" #Where all data live
    outFolder: str = 'C:/Users/zshawver/OneDrive - Dubin Consulting/Coding Sandbox/Juror Sampling'
    fn: str = "StateFarm_PreStim_9_16-17_2024.xlsx"

    sheetName: str = "labeledData"
    jurorID: str = 'jurorID'
    dv: str = 'DV_1Plaintiff_0Def'
    plaintiff_label: str = "Plaintiff"
    defense_label: str = "Def"

    #Read in data sheet
    df = pd.read_excel(os.path.join(folder, fn),sheet_name=sheetName)
    #Drop the juror name and DV columns
    sampleDF = df.drop(columns = [jurorID,dv])

    #Prepare Juror lists
    #Use original df with juror name and dv intact
    jurors, plaintiffJurors, defenseJurors = prepare_juror_lists(df, dv, jurorID,plaintiff_label,defense_label)

    #Prepare list of IV combinations
    #Returns all 2, 3, and 4-way combinations of IVs
    #Use df with juror name and dv removed
    iv_combos = prepare_iv_lists(sampleDF)

    singleIVs = [(col,) for col in sampleDF] #Individual IVs

    N = 12 #Sample size
    iterations = 100 #Number of samples
    for i in range(10):
        #Prepare Juror samples
        jurorSamples, plaintiffSamples, defenseSamples = prepare_random_samples_of_jurors(N,iterations,plaintiffJurors,defenseJurors)
        print(len(defenseSamples))

        #Combine samples & IVs
        # sample_ivs = prepare_sample_ivs_list(plaintiffSamples,iv_combos)
        # sample_ivs = prepare_sample_ivs_list(defenseSamples,iv_combos)
        sample_ivs = prepare_sample_ivs_list(plaintiffSamples, singleIVs)
        # sample_ivs = prepare_sample_ivs_list(defenseSamples, singleIVs)


        #Count iv level combinations for sample --> Parallel
        start = time.time()
        counts_parallel = count_iv_combos_parallel(sample_ivs,1400)
        end = time.time()
        total = end-start
        print(f"Counts took {total:.2f} seconds")

        start = time.time()
        counts_clean = [count for count in counts_parallel if len(count)>0]
        end = time.time()
        total = end-start
        print(f"Cleaning out empty Counters took {total:.2f} seconds")


        start = time.time()
        counts_clean = [count for count in counts_parallel if len(count)>0]
        # Split into chunks for parallel processing
        chunk_size = 1000 #len(counts_clean) // 10  # Adjust chunk size based on your environment
        chunks = [counts_clean[i:i + chunk_size] for i in range(0, len(counts_clean), chunk_size)]

        # Combine counters in parallel
        combined_counters = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(combine_counters, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                combined_counters.append(future.result())
        end = time.time()
        total = end-start
        print(f"Combining Counters took {total:.2f} seconds")

        start = time.time()
        # Initialize a defaultdict to hold the final counts
        final_count_dict = defaultdict(int)

        # Loop through each Counter in your combined_counters list
        for counter in combined_counters:
            for key, count in counter.items():
                final_count_dict[key] += count  # Accumulate counts

        # Convert the defaultdict to a regular dictionary if desired
        final_count_dict = dict(final_count_dict)
        end = time.time()
        total = end-start
        print(f"Pulling together final counts took {total:.2f} seconds")

        #Write the dictionary to a DF
        outDF = pd.DataFrame(final_count_dict.items(),columns = ["IV_Levels","Count"])

        #Filter out any values < 75th percentile
        # Q3 = outDF['Count'].quantile(.95)
        # outDF = outDF[outDF['Count'] > Q3]

        #Calculate Date & Time for file name
        date_time = datetime.now().isoformat()
        date_time = re.sub(r':[0-9]{2}\.[0-9]+', '', date_time)
        date_time = re.sub(r'T', '_', date_time)
        date_time = re.sub(r':', '-', date_time)

        #Prepare file name
        outFN = f"{date_time}_Plaintiffcounts.csv"
        #Write filtered file to Excel
        outDF.to_csv(os.path.join(outFolder,"Plaintiff",outFN), index = False)
