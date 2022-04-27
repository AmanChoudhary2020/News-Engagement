import pandas as pd
import numpy as np

from ast import literal_eval
from tqdm import tqdm

from sklearn.svm import LinearSVR
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from nltk import WordPunctTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

import pickle
from argparse import ArgumentParser

def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """

    if (metric == "f1-score"):
        return metrics.f1_score(y_true,y_pred)    
    elif (metric == "accuracy"):
        return metrics.accuracy_score(y_true,y_pred)
    elif (metric == "auroc"):
        return metrics.roc_auc_score(y_true,y_pred)
    elif (metric == "precision"):
        return metrics.precision_score(y_true,y_pred)
    elif (metric == "sensitivity"):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp/(tp + fn)
        return sensitivity
    elif (metric == "specificity"):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn+fp)
        return specificity
    elif (metric == "r-squared"):
        return metrics.r2_score(y_true, y_pred)

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """

    skf = StratifiedKFold(n_splits=k)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train,y_train)
        
        if (metric == "auroc"):
            y_pred = clf.decision_function(X_test)

        else:
            y_pred = clf.predict(X_test)
        
        score = performance(y_test,y_pred,metric)
        scores.append(score)
    
    return np.array(scores).mean()

    #scores = cross_val_score(clf, X, y, scoring=metric, cv=k)
    #return sum(scores)/len(scores) #average score from k-fold cross validation
    

def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True):

    """Search for hyperparameters of linear SVM with best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """

    avg_scores = []
    for c in C_range:
        #clf = LinearSVR(loss=loss,dual=dual,C=c,random_state=42)
        #clf = Ridge(alpha=c, random_state=42)
        #clf = LinearSVC(penalty=penalty,loss=loss,dual=dual,C=c,random_state=42)
        #clf = SVC(kernel='rbf',gamma = 0.5, C=c,random_state=42)
        clf = LogisticRegression(penalty="l2", C=c, random_state=42)
        avg = cv_performance(clf,X,y,k,metric)
        avg_scores.append(avg)

    max_index = avg_scores.index(max(avg_scores))
    return C_range[max_index]

def generate_train_test(df, label_data_col, label_data_normalize_col, train_data_col, feature_type, binary, remove_zero_values = True):
    '''
    Create training and test data sets
    df: post_id, author, upvotes, post_word_count, comment_body, article_content, headline_content,
    content_clean, content_stem, article_content_clean, headline_content_clean
    '''
    
    if (remove_zero_values):
        df = df.loc[df[label_data_col] > 0]
    labels = df[label_data_normalize_col].to_numpy()

    if binary:
        #split labels
        split = np.percentile(labels, 50)
        labels_copy = labels.copy()
        labels_copy[labels <= split] = 0
        labels_copy[labels > split] = 1
        labels = labels_copy

    if feature_type == "words":
        #documents is the array containing article text
        documents = []
        num_rows = df.shape[0]
        count = 0
        for index, row in tqdm(df.iterrows(), total=num_rows):
            count += 1
            #print(count)
            article_content_arr = literal_eval(row[train_data_col])        
            concatenated_article_string = ""
            for word in article_content_arr:
                concatenated_article_string += (word + " ")
            documents.append(concatenated_article_string)
        
        tokenizer = WordPunctTokenizer()

        cv = CountVectorizer(min_df=0.001, max_df=0.50, tokenizer=tokenizer.tokenize, lowercase=True) #Add stopwords, tokenizer, min_df, max_df
        X = cv.fit_transform(documents)

    elif feature_type == "topics":
        #documents is the array containing article text
        documents = []
        num_rows = df.shape[0]
        
        topic_model_file = 'best_lda_model_and_vocab.pkl'
        topic_model, topic_model_vocab = pickle.load(open(topic_model_file, 'rb'))

        for index, row in tqdm(df.iterrows(), total=num_rows):     
            article_content_arr = literal_eval(row[train_data_col])        
            concatenated_article_string = ""
            for word in article_content_arr:
                concatenated_article_string += (word + " ")
            documents.append(concatenated_article_string)
    
        cv = CountVectorizer(vocabulary=topic_model_vocab)
        data_text_dtm = cv.fit_transform(documents)

        # confirm that text matrix has same shape as original vocabulary
        assert data_text_dtm.shape[1] == len(topic_model_vocab)
        topic_probs = topic_model.fit_transform(data_text_dtm) 
        
        X = topic_probs

    data_train, data_test, labels_train, labels_test = train_test_split(X.toarray(), labels, test_size=0.20, random_state=42)
    return data_train, data_test, labels_train, labels_test

def main():
    parser = ArgumentParser()
    parser.add_argument('feature_type')
    args = vars(parser.parse_args())
    data_file = args['feature_type']

    df = pd.read_csv('../metrics_data/data.tsv', sep='\t', low_memory=False)

    label_data_col_arr = ['SS_Article_Text', 'Jaccard_Coef_Article', 'SS_Article_Headline', 'Jaccard_Coef_Headline']    
    label_data_normalize_col_arr = ['SS_Article_Text_Log_Transform', 'Jaccard_Coef_Article_Log_Transform', 'SS_Article_Headline_Log_Transform', 'Jaccard_Coef_Headline_Log_Transform']
    train_data_col_arr = ['article_content_clean', 'headline_content_clean']

    for train_data_col in train_data_col_arr:
        if (train_data_col == 'article_content_clean'):
            start = 0
            end = 2
        else:
            start = 2
            end = 4

        for index in range(start, end):
            print(label_data_col_arr[index])
            print(label_data_normalize_col_arr[index])
            print(train_data_col)

            print("Splitting training and testing datasets...")
            feature_type = data_file
            binary = True
            data_train, data_test, labels_train, labels_test = generate_train_test(df, label_data_col_arr[index], label_data_normalize_col_arr[index], train_data_col, feature_type, binary)
            
            '''
            LinearSVR:
            
            print("Selecting optimal C parameter...")
            
            C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            optimal_c = select_param_linear(data_train, labels_train, 5, 'r-squared', C_range, 'epsilon_insensitive', penalty="l2", dual=True) # note: penalty not being used with LinearSVR
            
            print("Optimal C Parameter:")
            print(optimal_c)
            
            model = LinearSVR(loss='epsilon_insensitive',dual=True,C=optimal_c,random_state=42)
            '''

            '''
            Ridge:
            
            
            print("Selecting optimal alpha parameter...")
            alpha_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
            
            optimal_alpha = select_param_linear(data_train, labels_train, 5, 'r-squared', alpha_range, 'epsilon_insensitive', penalty="l2", dual=True) # note: penalty not being used with LinearSVR
            
            print("Optimal Alpha Parameter:")
            print(optimal_alpha)
            
            model = Ridge(alpha = optimal_alpha, random_state=42)
            '''

            '''
            LinearSVC/SVC:
            
            
            print("Selecting optimal C parameter...")
            
            C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            optimal_c = select_param_linear(data_train, labels_train, 5, 'accuracy', C_range, 'hinge', penalty="l2", dual=True) 
            
            print("Optimal C Parameter:")
            print(optimal_c)
            
            #model = LinearSVC(penalty='l2',loss='hinge',dual=True,C=optimal_c,random_state=42)
            model = SVC(kernel='rbf',C=optimal_c, gamma=0.5, random_state=42)
            '''
            
            '''
            Logistic Regression:
            '''
            
            print("Selecting optimal C parameter...")
            
            C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            optimal_c = select_param_linear(data_train, labels_train, 5, 'accuracy', C_range, 'hinge', penalty="l2", dual=True) 
            
            print("Optimal C Parameter:")
            print(optimal_c)
            
            #model = LinearSVC(penalty='l2',loss='hinge',dual=True,C=optimal_c,random_state=42)
            model = LogisticRegression(penalty="l2", C=optimal_c, random_state=42)
            
            #TODO
            #logistic regression
            #possible parameter-> word mapping
            #multiclass
            #LSTM 
            #look at num of 0's and 1's in binary

            #----- FIT -----#
            model.fit(data_train,labels_train)
            score = model.score(data_test,labels_test)
            
            print("score:")
            print(score)   
            print('\n')

if __name__ == '__main__':
    main()