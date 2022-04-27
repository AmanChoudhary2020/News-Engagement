'''
Takes existing reddit comment data, adds one column with article topic 
(for each comment, takes the article from the corresponding post on which the comment was created)

Overwrites to metrics_data/semantic_sim.tsv

dependencies:
- pandas
'''

import os
import pandas as pd
from tqdm import tqdm

def add_topics(semantic_sim_file_path, topics_file_path, file):
    semantic_sim = pd.read_csv(semantic_sim_file_path, sep='\t', low_memory=False)
    topics = pd.read_csv(topics_file_path, sep='\t', low_memory=False)
    num_rows = semantic_sim.shape[0]

    topics_arr = []
    
    for index, row in tqdm(semantic_sim.iterrows(), total=num_rows):
        submission_id = row['post id']
        thing = False
        for index1, row1 in topics.iterrows():
            if row1['post id'] == submission_id:
                topics_arr.append(row1['doc_topic'])
                thing = True
                break
        
        if thing == False:
            print(row['post id'])
        

    semantic_sim["topic"] = topics_arr
    semantic_sim.to_csv(file,mode='a', index=False, sep='\t')

def main():
    if os.path.exists("semantic_sim.tsv"):
        os.remove("semantic_sim.tsv")

    add_topics('../metrics_data/semantic_sim.tsv', '../metrics_data/data_topics_repeats.tsv', 'semantic_sim.tsv')

if __name__ == "__main__":
    main()
