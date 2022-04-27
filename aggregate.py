'''
(1) Accumulates each month's Reddit post data to one TSV/Excel file, from 06/2020 to 09/2020. 
(2) Adds two more columns for each Reddit post with cleaned and tokenized article content

Note: these two columns were later deleted from data
TSV/Excel files found in raw_data/data_clean
'''

import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer

def clean_text(df, col='content'):
    df_copy = df.copy()
    
    # lower the text
    df_copy['preprocessed_' + col] = df_copy[col].str.lower()
    
    # filter out stop words
    stop_words = nltk.corpus.stopwords.words('english')
    df_copy['preprocessed_' + col] = df_copy['preprocessed_' + col].apply(lambda row: ' '.join([word for word in str(row).split() if (not word in stop_words)]))
    
    # tokenize the article text
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
    df_copy['tokenized_' + col] = df_copy['preprocessed_' + col].apply(lambda row: tokenizer.tokenize(row))
    
    return df_copy

def accumulate_documents():
    june = pd.read_excel('guardian_06_07.xlsx')
    june_clean = clean_text(june)
    july = pd.read_excel('guardian_07_08.xlsx')
    july_clean = clean_text(july)
    aug = pd.read_excel('guardian_08_09.xlsx')
    aug_clean = clean_text(aug)
    frames = [june_clean,july_clean,aug_clean]
    data_clean = pd.concat(frames)
    data_clean.to_excel('data_clean.xlsx')

def main():
    accumulate_documents()

if __name__ == "__main__":
    main()
