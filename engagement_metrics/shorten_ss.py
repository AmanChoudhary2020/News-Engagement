'''
For sake of making data more convenient to look at, removes all article and comment text, 
only saves post id, author, Jaccard_Coef_Article, Jaccard_Coef_Headline, SS_Article_Text, 
and SS_Article_Headline metrics.

Output can be found at metrics_data/semantic_sim_short.tsv

'''
#from pydoc_data.topics import topics
import pandas as pd
from tqdm import tqdm
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
from ast import literal_eval
import os
import json

def shorten_ss(input_file, output_file):
    semantic_data = pd.read_csv(input_file, sep='\t', low_memory=False)
    semantic_data = semantic_data[["post id", "author", "Jaccard_Coef_Article", "Jaccard_Coef_Headline", "SS_Article_Text", "SS_Article_Headline"]]
    semantic_data.to_csv(output_file, mode='a', index=False, sep='\t')

def main():
    if os.path.exists("semantic_sim_short.tsv"):
        os.remove("semantic_sim_short.tsv")

    shorten_ss("../metrics_data/semantic_sim.tsv", "semantic_sim_short.tsv")

if __name__ == "__main__":
    main()
