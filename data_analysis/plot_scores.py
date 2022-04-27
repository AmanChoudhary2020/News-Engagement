'''
Plot Jaccard similarity/semantic similarity for all comments on a histogram
'''

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_score(scores_arr, output_file, xlab, ylab, title):
    '''
    scores_arr contains all scores 
    output_file specifies file to save histogram chart
    '''
    plt.hist(scores_arr, bins='auto', align='mid')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

    plt.savefig(output_file)
    plt.cla()


def main():
    scores = pd.read_csv('../metrics_data/semantic_sim.tsv', sep='\t', low_memory=False)

    plot_score(scores['Jaccard_Coef_Article'], 'jaccard_article_scores.png', 'Jaccard Score Article Text', '# of Occurrences', 'Jaccard Score Article Text')
    plot_score(scores['SS_Article_Text'], 'semantic_article_scores.png', 'Semantic Score Article Text', '# of Occurrences', 'Semantic Score Article Text')
    plot_score(scores['Jaccard_Coef_Headline'], 'jaccard_headline_scores.png', 'Jaccard Score Headline Text', '# of Occurrences', 'Jaccard Score Headline Text')
    plot_score(scores['SS_Article_Headline'], 'semantic_headline_scores.png', 'Semantic Score Article Text', '# of Occurrences', 'Semantic Score Headline Text')

if __name__ == '__main__':
    main()