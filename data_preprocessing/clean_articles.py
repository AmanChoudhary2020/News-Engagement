'''
Takes existing article data, adds two new columns for each article with cleaned article content 
and cleaned article headline

Clean text is defind as: lowercase, stop words removed, punctuation removed

Saves to processed_data/clean_articles.tsv

dependencies:
- pandas
- nltk
'''

import pandas as pd
from tqdm import tqdm
import nltk
import os

def clean_articles(articles_file_path, file):
    articles_data = pd.read_excel(articles_file_path)

    num_rows = articles_data.shape[0]

    cleaned_article_text = []
    cleaned_headline_text = []

    for index, row in tqdm(articles_data.iterrows(), total=num_rows):
        cleaned_article_text.append(clean_text(row['content']))
        cleaned_headline_text.append(clean_text(row['title']))
    
    articles_data['content_clean'] = cleaned_article_text
    articles_data['title_clean'] = cleaned_headline_text
    articles_data.to_csv(file, mode='a', index=False, sep='\t')

def clean_text(token):
    #lowercase, remove stop words, remove punctuation

    words_doc1 = set(token.lower().split())
    stop_words = nltk.corpus.stopwords.words('english')
    result = []

    punc = '''!()-[]{};:'“”’"\,<>./?@#$%^&*_~'''

    for word1 in words_doc1:
        if not word1.isnumeric():
            for ele in word1:
                if ele in punc:
                    word1 = word1.replace(ele, "")
            if (word1 not in stop_words):
                result.append(word1)

    return set(result)

def main():
    if os.path.exists("clean_articles.tsv"):
        os.remove("clean_articles.tsv")

    clean_articles('../raw_data/data_clean.xlsx','clean_articles.tsv')

if __name__ == "__main__":
    main()
