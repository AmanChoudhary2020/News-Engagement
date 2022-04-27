'''
Takes existing (cleaned) Reddit comments data, adds one new column for each comment with 
stemmed comment text

Saves to processed_data/stem_clean_comments.tsv

dependencies:
- pandas
- PorterStemmer
'''

import pandas as pd
from tqdm import tqdm
from nltk.stem import PorterStemmer
from ast import literal_eval
import os

def stem_comments(clean_comments_file_path, file):
    ps = PorterStemmer()
    stem_comments = pd.read_csv(clean_comments_file_path, sep='\t', low_memory=False)

    num_rows = stem_comments.shape[0]
    content_stem_final = []

    for index, row in tqdm(stem_comments.iterrows(), total=num_rows):
        clean_words = literal_eval(row["content_clean"])
        result = []

        for word in clean_words:
            word = ps.stem(word)
            result.append(word)
        
        content_stem_final.append(result)

    stem_comments['content_stem'] = content_stem_final
    stem_comments.to_csv(file, mode='a', index=False, sep='\t')

def main():
    if os.path.exists("stem_clean_comments.tsv"):
        os.remove("stem_clean_comments.tsv")

    stem_comments('../processed_data/clean_comments.tsv', 'stem_clean_comments.tsv')

if __name__ == "__main__":
    main()
