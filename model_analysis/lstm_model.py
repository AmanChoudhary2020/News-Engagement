from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from linear_models import generate_train_test
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from argparse import ArgumentParser
import numpy as np

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
            binary = False
            data_train, data_test, labels_train, labels_test = generate_train_test(df, label_data_col_arr[index], label_data_normalize_col_arr[index], train_data_col, feature_type, binary)
            #data_train.sort_indices()
            #data_test.sort_indices()
            #print(data_train.shape)
            #data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], 1)
            #data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], 1)

            model = Sequential()
            model.add(LSTM(units=50,return_sequences=True,input_shape=(data_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mse'])
            model.fit(data_train,labels_train,epochs=1,batch_size=32)
            
            loss, mse = model.evaluate(data_test, labels_test)

            print("score:")
            print(loss)   
            print(mse)
            print('\n')


if __name__ == '__main__':
    main()