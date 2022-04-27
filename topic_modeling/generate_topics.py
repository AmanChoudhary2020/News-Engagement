"""
Generate topics for text data using LDA.
(1) Clean text.
(2) Test varying numbers of topics w/ LDA, compute coherence scores.
(3) Take model w/ best coherence score, re-run on text, write topics to file.

dependencies:
- pickle
- numpy
- pandas
- nltk
- scikit-learn
- tmtoolkit (for coherence scores)
"""
import pickle
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from nltk import WordPunctTokenizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tmtoolkit.topicmod.evaluate import metric_coherence_mimno_2011

def get_top_words_per_topic(lda_model, num_topics, cv_vocab):
    """
    Get all words per topic with highest conditional probability.

    :param lda_model:
    :param num_topics:
    :param cv_vocab:
    :return: top_word_per_topic
    """
    top_words_per_topic = []
    top_k = 20
    for i in range(num_topics):
        # compute highest P(word | topic)
        topic_probs_i = pd.Series(lda_model.components_[i, :], index=cv_vocab).sort_values(ascending=False)
        top_words_per_topic_i = topic_probs_i.index[:top_k].tolist()
        top_words_per_topic.append(top_words_per_topic_i)
    top_words_per_topic = pd.DataFrame([[i, words_i] for i, words_i in enumerate(top_words_per_topic)],
                                       columns=['topic', 'topic_words'])
    return top_words_per_topic

def compute_coherence_scores(dtm, num_topic_list=[10, 20, 30, 50, 100]):
    """
    Compute coherence scores for varying numbers of topics.

    :param dtm:
    :param num_topic_list:
    :return: coherence_vals
    """
    coherence_vals = []
    for num_topics in num_topic_list:
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=123)
        lda_model.fit(dtm)
        topic_probs = lda_model.components_ / lda_model.components_.sum(axis=1).reshape(-1, 1)
        coh = metric_coherence_mimno_2011(dtm=dtm, topic_word_distrib=topic_probs, top_n=50)
        coherence_vals.append([num_topics, np.median(coh)])
    coherence_vals = pd.DataFrame(coherence_vals, columns=['num_topics', 'coherence'])
    coherence_vals.sort_values('coherence', ascending=False, inplace=True)
    return coherence_vals

def main():
    parser = ArgumentParser()
    parser.add_argument('data_file')
    args = vars(parser.parse_args())
    data_file = args['data_file']

    ## load data
    data = pd.read_excel(data_file, index_col=0)
    # remove duplicate articles
    data.drop_duplicates('content', inplace=True)
    # clean text: tokenize, stem all words, convert back to string
    tokenizer = WordPunctTokenizer()
    stemmer = PorterStemmer()
    clean_text = data.loc[:, 'content'].apply(lambda x: ' '.join(list(map(lambda y: stemmer.stem(y), tokenizer.tokenize(x.lower())))))
    print(clean_text)
    ## transform data
    # compute document term matrix
    en_stops = stopwords.words('english')
    cv = CountVectorizer(min_df=0.001, max_df=0.5, tokenizer=tokenizer.tokenize, lowercase=True, stop_words=en_stops)
    dtm = cv.fit_transform(clean_text)
    cv_vocab = list(sorted(cv.vocabulary_, key=cv.vocabulary_.get))

    ## test topic modeling
    # find best number of topics w/ coherence score
    num_topic_list = [10, 20, 30, 50, 100]
    coherence_vals = compute_coherence_scores(dtm, num_topic_list=num_topic_list)
    ## save to file for records
    coherence_vals.to_csv('data_topic_coherence_vals.csv', index=False)

    ## convert data to topics
    # re-fit model
    best_num_topics = coherence_vals.loc[:, 'num_topics'].iloc[0]
    best_lda_model = LatentDirichletAllocation(n_components=best_num_topics, random_state=123)
    best_lda_model.fit(dtm)
    print(clean_text)
    # save model + vocab for later
    pickle.dump((best_lda_model, cv_vocab), open('best_lda_model_and_vocab.pkl', 'wb'))
    # collect top words per topic
    top_words_per_topic = get_top_words_per_topic(best_lda_model, best_num_topics, cv_vocab)
    # save to file
    top_words_per_topic.to_csv('best_lda_model_topic_words.tsv', sep='\t', index=False)
    # assign 1 topic per document
    topic_probs = best_lda_model.transform(dtm)
    doc_topics = topic_probs.argmax(axis=1)
    doc_topic_probs = topic_probs.max(axis=1)
    data = data.assign(**{
        'doc_topic': doc_topics,
        'doc_topic_prob': doc_topic_probs,
    })
    data.to_csv('data_topics.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()
