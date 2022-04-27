'''
Calculates Jaccard similarity between every comment and its corresponding article and 
article headline. Appends this data to end of comments TSV file.

Data can be found at metrics_data/comments_jaccard.tsv

dependencies:
- pandas
- numpy 
'''

import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np
import os

def jaccard_coef(tokens_1, tokens_2):
    # compute len(intersection of tokens) / len(union of tokens)
    # return coef

    # List the unique words in a document
    try:
        # Find the intersection of words list of doc1 & doc2
        intersection = list(set(tokens_1).intersection((tokens_2)))
        # Find the union of words list of doc1 & doc2
        union = list(set(tokens_1).union((tokens_2)))
 
        # Calculate Jaccard similarity score
        # using length of intersection set divided by length of union set
        return float(len(intersection)) / len(union)
    except NameError:
        return -1

def jaccard_coef_comments(comments_file, articles_file, file):
    comments_data = pd.read_csv(comments_file, sep='\t', low_memory=False)
    articles_data = pd.read_csv(articles_file, sep='\t', low_memory=False)
    scores1 = []
    scores2 = []

    num_rows = comments_data.shape[0]

    for index, row in tqdm(comments_data.iterrows(), total=num_rows):
        if (row['post id'] != "post id"):
            a = articles_data.loc[articles_data['post id'] == row['post id']]
            article_text = a['content_stem'].values
            headline_text = a['headline_stem'].values

            if ((article_text.shape[0] != 0) and (headline_text.shape[0] != 0)) :
                coef = jaccard_coef(literal_eval(row['content_stem']), literal_eval(article_text[0]))
                scores1.append(coef)
                coef2 = jaccard_coef(literal_eval(row['content_stem']), literal_eval(headline_text[0]))
                scores2.append(coef2)
        else:
            scores1.append(np.nan)
            scores2.append(np.nan)
            
    comments_data["Jaccard_Coef_Article"] = scores1
    comments_data["Jaccard_Coef_Headline"] = scores2

    comments_data.to_csv(file,mode='a', index=False, sep='\t')


def main():
    if os.path.exists("comments_jaccard.tsv"):
        os.remove("comments_jaccard.tsv")

    jaccard_coef_comments("../processed_data/stem_clean_comments.tsv","../processed_data/stem_clean_articles.tsv", "comments_jaccard.tsv")

if __name__ == "__main__":
    main()
