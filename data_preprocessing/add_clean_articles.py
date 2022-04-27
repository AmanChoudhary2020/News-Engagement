'''
Takes existing reddit comment data, adds two new columns with article content and article headline 
(for each comment, takes the article from the corresponding post on which the comment was created)

Saves to raw_data/comments.tsv

dependencies:
- pandas
'''

import os
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

def add_articles(log_file_path, articles_file_path, file):
    log_data = pd.read_csv(log_file_path, sep='\t', low_memory=False)
    clean_articles = pd.read_csv(articles_file_path, sep='\t', low_memory=False)
    num_rows = log_data.shape[0]

    content = []
    headline = []
    
    for index, row in tqdm(log_data.iterrows(), total=num_rows):
        a = clean_articles.loc[clean_articles['post id'] == row['post id']]
        article_text = literal_eval(a['content_stem'].values[0])
        header_text = literal_eval(a['headline_stem'].values[0])
        content.append(article_text)
        headline.append(header_text)

    log_data["article_content_clean"] = content
    log_data["headline_content_clean"] = headline

    log_data.to_csv(file,mode='a', index=False, sep='\t')

def main():
    if os.path.exists("comments.tsv"):
        os.remove("comments.tsv")

    add_articles('../metrics_data/log_data.tsv', '../processed_data/stem_clean_articles.tsv', 'data.tsv')

if __name__ == "__main__":
    main()
