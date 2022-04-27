'''
Creates and saves two dictionaries with following key/value pairing:

article_embeddings.json:
key - article post id
value - article content embedding (with cleaned article content text)

article_headline_embeddings.json:
key - article post id
value - article headline embedding (with cleaned article headline text)

dependencies:
- pandas
- json 
- pretrained word embeddings
'''

import pandas as pd
import json
from ast import literal_eval
from tqdm import tqdm
from semantic_similarity import get_average_embedding
import os

def generate_article_embeddings(articles_file, article_body_embedding, article_headline_embedding, embeddings):
    articles_data = pd.read_csv(articles_file, sep='\t', low_memory=False)
    result1 = {}
    result2 = {}

    num_rows = articles_data.shape[0]

    for index, row in tqdm(articles_data.iterrows(), total=num_rows):
        tokens1 = literal_eval(row['content_clean'])
        tokens2 = literal_eval(row['title_clean'])

        result1[row['post id']] = get_average_embedding(tokens1, embeddings).tolist()
        result2[row['post id']] = get_average_embedding(tokens2, embeddings).tolist()
    
    with open(article_body_embedding, 'w') as outfile:
        json.dump(result1, outfile, indent=4)
    
    with open(article_headline_embedding, 'w') as outfile:
        json.dump(result2, outfile, indent=4)

def main():
    word_vecs = pd.read_pickle("word_vecs.pkl")
    
    if os.path.exists("article_body_embeddings.json"):
        os.remove("article_body_embeddings.json")

    if os.path.exists("article_headline_embeddings.json"):
        os.remove("article_headline_embeddings.json")

    generate_article_embeddings("../processed_data/clean_articles.tsv", "article_body_embeddings.json", "article_headline_embeddings.json", word_vecs)

if __name__ == "__main__":
    main()

