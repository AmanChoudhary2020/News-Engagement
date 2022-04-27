from plot_scores import plot_score
import pandas as pd

def main():
    scores = pd.read_csv('../metrics_data/normalized_data.tsv', sep='\t', low_memory=False)

    plot_score(scores['Jaccard_Coef_Article_Normalized'], 'jaccard_normalized_article_scores.png', 'Jaccard Score Normalized Article Text', '# of Occurrences', 'Jaccard Score Normalized Article Text')
    plot_score(scores['SS_Article_Text_Normalized'], 'semantic_normalized_article_scores.png', 'Semantic Score Normalized Article Text', '# of Occurrences', 'Semantic Score Normalized Article Text')
    plot_score(scores['Jaccard_Coef_Headline_Normalized'], 'jaccard_normalized_headline_scores.png', 'Jaccard Score Normalized Headline Text', '# of Occurrences', 'Jaccard Score Normalized Headline Text')
    plot_score(scores['SS_Article_Headline_Normalized'], 'semantic_normalized_headline_scores.png', 'Semantic Score Normalized Article Text', '# of Occurrences', 'Semantic Score Normalized Headline Text')

if __name__ == '__main__':
    main()