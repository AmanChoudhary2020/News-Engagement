from plot_scores import plot_score
import pandas as pd

def main():
    scores = pd.read_csv('../metrics_data/log_data.tsv', sep='\t', low_memory=False)

    plot_score(scores['Jaccard_Coef_Article_Log_Transform'], 'jaccard_log_transform_article_scores.png', 'Jaccard Score Log Tranform Article Text', '# of Occurrences', 'Jaccard Score Log Tranform Article Text')
    plot_score(scores['SS_Article_Text_Log_Transform'], 'semantic_log_transform_article_scores.png', 'Semantic Score Log Tranform Article Text', '# of Occurrences', 'Semantic Score Log Tranform Article Text')
    plot_score(scores['Jaccard_Coef_Headline_Log_Transform'], 'jaccard_log_transform_headline_scores.png', 'Jaccard Score Log Tranform Headline Text', '# of Occurrences', 'Jaccard Score Log Tranform Headline Text')
    plot_score(scores['SS_Article_Headline_Log_Transform'], 'semantic_log_transform_headline_scores.png', 'Semantic Score Log Tranform Article Text', '# of Occurrences', 'Semantic Score Log Tranform Headline Text')

if __name__ == '__main__':
    main()