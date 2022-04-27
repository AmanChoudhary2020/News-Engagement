'''
Normalize scores
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize_scores(data):
    scaler = StandardScaler()
    data= scaler.fit_transform(data)
    return data

def prepare(data):
    result = []
    for ele in data:
        result.append([ele])
    return result

def undo(data):
    result = []
    for ele in data:
        result.append(ele[0])
    return result

def normalize(scores, col, newcol):
    data = scores[col].to_list()
    data = prepare(data)
    data_normalized = undo(normalize_scores(data))
    scores[newcol] = data_normalized

def main():
    scores = pd.read_csv('../metrics_data/semantic_sim.tsv', sep='\t', low_memory=False)
    
    normalize(scores, 'Jaccard_Coef_Article', 'Jaccard_Coef_Article_Normalized')
    normalize(scores, 'SS_Article_Text', 'SS_Article_Text_Normalized')
    normalize(scores, 'Jaccard_Coef_Headline', 'Jaccard_Coef_Headline_Normalized')
    normalize(scores, 'SS_Article_Headline', 'SS_Article_Headline_Normalized')
    
    scores.to_csv('normalized_data.tsv', mode='a', index=False, sep='\t')

if __name__ == '__main__':
    main()