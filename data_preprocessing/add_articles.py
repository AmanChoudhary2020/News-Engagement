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

def add_articles(comments_file_path, articles_file_path, file):
    comments_data = pd.read_csv(comments_file_path, sep='\t', low_memory=False)
    articles_data = pd.read_excel(articles_file_path)
    num_rows = comments_data.shape[0]

    content = []
    headline = []
    
    for index, row in tqdm(comments_data.iterrows(), total=num_rows):
        a = articles_data.loc[articles_data['post id'] == row['post id']]
        article_text = a['content'].values
        header_text = a['title'].values
        content.append(article_text)
        headline.append(header_text)

    comments_data["article_content"] = content
    comments_data["headline_content"] = headline

    comments_data.to_csv(file,mode='a', index=False, sep='\t')

def main():
    if os.path.exists("comments.tsv"):
        os.remove("comments.tsv")

    add_articles('../raw_data/comments.csv', '../raw_data/data_clean.xlsx', 'comments.tsv')

if __name__ == "__main__":
    main()
