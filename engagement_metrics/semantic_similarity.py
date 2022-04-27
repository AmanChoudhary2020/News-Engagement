'''
Calculates semantic similarity between every comment and its corresponding article and 
article headline. Appends this data to end of comments TSV file.

Data can be found at metrics_data/semantic_sim.tsv

dependencies:
- pandas
- numpy 
- json
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

def get_average_embedding(tokens, embeddings):
    # get embedding for each token
    # "stack" embeddings into matrix (rows = tokens, columns = embedding dimensions)
    # compute average along each dimension (per-column)
    # return average embedding

    arr = []

    for words in tokens:
        test_word = words
        # this creates a 300-dimensional vector
        try:
            test_word_vec = embeddings.loc[test_word, :]
            arr.append(test_word_vec)
        except:
            pass
    
    if (len(arr) == 0):
        print(tokens)
        return np.zeros(300)
    
    num_arr = np.array(arr)
    return num_arr.mean(axis=0)

def semantic_sim(tokens_1, tokens_2, embeddings):
    # compute cosine similarity between embeddings
    # return similarity

    average_embedding_1 = get_average_embedding(tokens_1, embeddings)
    #average_embedding_2 = get_average_embedding(tokens_2, embeddings)
    average_embedding_2 = tokens_2

    if((norm(average_embedding_1) == 0) or (norm(average_embedding_2) == 0)):
        print(norm(average_embedding_1))
        return 0
        
    #print(dot(average_embedding_1,average_embedding_2)/(norm(average_embedding_1) * norm(average_embedding_2)))
    return 1 - distance.cosine(average_embedding_1, average_embedding_2)

def sim(comments_file, article_embeddings, article_headline_embeddings, output_file_name, embeddings):
    comments_data = pd.read_csv(comments_file, sep='\t', low_memory=False)
    result1 = []
    result2 = []

    num_rows = comments_data.shape[0]

    # use preprocessed article text
    for index, row in tqdm(comments_data.iterrows(), total=num_rows):
        sim = semantic_sim(literal_eval(row['content_clean']), article_embeddings[row['post id']], embeddings)
        sim2 = semantic_sim(literal_eval(row['content_clean']), article_headline_embeddings[row['post id']], embeddings)
        result1.append(sim)
        result2.append(sim2)

    comments_data["SS_Article_Text"] = result1
    comments_data["SS_Article_Headline"] = result2

    comments_data.to_csv(output_file_name, mode='a', index=False, sep='\t')

def main():
    word_vecs = pd.read_pickle("word_vecs.pkl")

    if os.path.exists("fake_semantic_sim.tsv"):
        os.remove("fake_semantic_sim.tsv")

    with open('article_embeddings.json') as json_file:
        article_embeddings = json.load(json_file)

    with open('article_headline_embeddings.json') as json_file:
        article_headline_embeddings = json.load(json_file)

    sim("comments_jaccard.tsv", article_embeddings, article_headline_embeddings, "fake_semantic_sim.tsv", word_vecs)

if __name__ == "__main__":
    main()
