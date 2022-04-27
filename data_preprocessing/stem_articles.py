'''
Takes existing (cleaned) Reddit articles data, adds two new columns for each article with 
stemmed article text and stemmed article headline

Saves to processed_data/stem_clean_articles.tsv

dependencies:
- pandas
- PorterStemmer
'''

import pandas as pd
from tqdm import tqdm
from nltk.stem import PorterStemmer
from ast import literal_eval
import os

def stem_comments(articles_file_path, file):
    ps = PorterStemmer()
    stem_articles = pd.read_csv(articles_file_path, sep='\t', low_memory=False)

    num_rows = stem_articles.shape[0]
    content_stem_final = []
    headline_stem_final = []
    
    for index, row in tqdm(stem_articles.iterrows(), total=num_rows):        
        content_clean = literal_eval(row['content_clean'])
        headline_clean = literal_eval(row['title_clean'])

        content_stem = []
        headline_stem = []

        for word in content_clean:
            word = ps.stem(word)
            content_stem.append(word)
        for word in headline_clean:
            word = ps.stem(word)
            headline_stem.append(word)

        content_stem_final.append(content_stem)
        headline_stem_final.append(headline_stem)

    stem_articles['content_stem'] = content_stem_final
    stem_articles['headline_stem'] = headline_stem_final        
    stem_articles.to_csv(file, mode='a', index=False, sep='\t')
    
def main():
    if os.path.exists("stem_clean_articles.tsv"):
        os.remove("stem_clean_articles.tsv")

    stem_comments('../processed_data/clean_articles.tsv', 'stem_clean_articles.tsv')

if __name__ == "__main__":
    main()
