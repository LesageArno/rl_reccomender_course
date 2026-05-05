# To run from the root: python -m UIR.Scripts.evaluationFuzzy

# To retrieve the files and folders
import os
from pathlib import Path

# To manipulate and gather the data
import pandas as pd

# To observe the data and the processing time
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Should we display the plot at runtime
SHOW = False

# Headers of the "headerless" txt (tsv) files  
COLS = ["Iteration", "Average jobs", "Average reward", "Average goal gap", "Average pref cov", "Average total levels req", "Average skills req unique", "Average skills fully covered", "Average skills missing unique", "Time"]

# Metric on which we want to evaluate versus iteration
EVALUATE_ON = ["Average jobs", "Average reward", "Average goal gap", "Average pref cov", "Average total levels req", "Average skills req unique", "Average skills fully covered", "Average skills missing unique", "Time"]

# Type of metric being maximised
METRIC = "Employability"

# Get working directory and extraction path
ROOT_PATH = Path(os.getcwd())
RESULTS_PATH = Path("UIR/resultsFuzzy")
EXTRACT_PATH = ROOT_PATH / RESULTS_PATH


## Gather the files into one big dataframe
df_list = []

# For each files in all subdirectory recursively
for path, folder, files in os.walk(EXTRACT_PATH):
    # If it is an empty folder, continue
    if files == []:
        continue
    # For each file in the subfolder
    for file in files:
        # If the file start by plot, then continue
        if file.startswith("plot"):
            continue
        
        # Get the metric, the method, the length of the sequence and the seed from the file name
        metric, method, k, seed = file.removesuffix(".txt").split("_")
        k = k.removeprefix("k")
        seed = seed.removeprefix("seed")
        
        # Read the txt (as tsv), attach the corresponding column and add metadata informations to have unique values during merge of dataframe
        df = pd.read_csv(Path(path) / file, sep = " ", header=None)
        df.columns = COLS
        df[["Metric"]] = metric
        df[["Method"]] = method
        df[["k"]] = int(k)
        df[["Seed"]] = int(seed) 
        
        # Add the processed dataframe into a list
        df_list.append(df)

# Merge all the dataframe
df = pd.concat(df_list, axis=0, ignore_index=True)

# Keep only the reward type we are interested in
filteredDF = df.loc[df["Metric"]==METRIC,:]

## Start the plots
begin = time.time()

# For each plot 
for i, evaluation in enumerate(EVALUATE_ON, start=1):
    # Create a subplot containing each sequence length from 2 to 5 (excluded)
    fig, axes = plt.subplots(1,3, constrained_layout = True)
    for k in range(2,5):
        # Plot the metrics over iteration and colour the methods. Confidence interval are computed as bootstrap using lineplot
        sns.lineplot(filteredDF.loc[filteredDF["k"]==k,:], x="Iteration", y=evaluation, hue="Method", ax=axes[(k+1)%3])
        axes[(k+1)%3].set_title(f"{k=}")
    
    # Modify the figure parameters (size and title)
    fig.suptitle(f"{evaluation} and Confidence Interval at 95% VS Iteration for {METRIC}")
    fig.set_size_inches(15.6, 8.7)
    
    # Save the figures
    plt.savefig(EXTRACT_PATH / "saved_fuzzy_results" / f"plot_{METRIC.lower()}_{'_'.join(evaluation.lower().split(' '))}.png", dpi=600)
    
    # Show advacement
    print(f"[{i}/{len(EVALUATE_ON)}] Iteration VS {evaluation}. Time from start: {time.time()-begin:.4f}s")
    
    # Display the plot if asked
    if SHOW:
        plt.show()
    

    