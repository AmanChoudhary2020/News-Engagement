'''
'''

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def log_transform(scores, col, newcol):
    data = scores[col].to_numpy()
    data = (data - data.min()) / (data.max() - data.min())
    smooth_val = data[data!=0].min() * 1e-1
    data = data + smooth_val
    data_log = np.log(data) 
    scores[newcol] = data_log

def main():
    scores = pd.read_csv('../metrics_data/normalized_data.tsv', sep='\t', low_memory=False)

    log_transform(scores, 'Jaccard_Coef_Article', 'Jaccard_Coef_Article_Log_Transform')
    log_transform(scores, 'SS_Article_Text', 'SS_Article_Text_Log_Transform')
    log_transform(scores, 'Jaccard_Coef_Headline', 'Jaccard_Coef_Headline_Log_Transform')
    log_transform(scores, 'SS_Article_Headline', 'SS_Article_Headline_Log_Transform')

    scores.to_csv('log_data.tsv', mode='a', index=False, sep='\t')

if __name__ == '__main__':
    main()