'''
Takes existing Reddit comments data, adds one new column for each comment with cleaned comment text.
Removes bots and other unhelpful comments.

Clean text is defind as: lowercase, stop words removed, punctuation removed

Saves to processed_data/clean_comments.tsv

dependencies:
- pandas
- nltk
'''

import pandas as pd
from tqdm import tqdm
import numpy as np
import nltk
import os

def clean_comments(comments_file_path, file):
    comments_data = pd.read_csv(comments_file_path, sep='\t', low_memory=False)
    num_rows = comments_data.shape[0]

    for index, row in tqdm(comments_data.iterrows(), total=num_rows):
        if (row['comment body'] == "[removed]" or row['comment body'] == "[deleted]"):
            comments_data = comments_data.drop(index)
        if (row['comment body'].startswith('>')):
            comments_data = comments_data.drop(index)
        if (row['author'] is not np.nan and (row['author'].lower() == "autotldr" or row['author'].lower() == "automoderator")):
            comments_data = comments_data.drop(index)
   
    cleaned_comment_text = []
    
    num_rows = comments_data.shape[0]

    for index, row in tqdm(comments_data.iterrows(), total=num_rows):
        cleaned_comment_text.append(clean_text(row['comment body']))

    comments_data["content_clean"] = cleaned_comment_text
    comments_data.to_csv(file, mode='a', index=False, sep='\t')

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
    if os.path.exists("clean_comments.tsv"):
        os.remove("clean_comments.tsv")

    clean_comments('../raw_data/comments.tsv','clean_comments.tsv')

if __name__ == "__main__":
    main()
