import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import collections

def per_topic(data1, data2):
    num_rows = data1.shape[0]
    per_topic = {}       
    size = 0 

    for index, row in tqdm(data1.iterrows(), total=num_rows):
        submission_id = row['post id']

        for index1, row1 in data2.iterrows():
            if row1['post id'] == submission_id:
                topic = row1['doc_topic']
                break

        arr = []
        arr.append(row['Jaccard_Coef_Article'])
        arr.append(row['Jaccard_Coef_Headline'])
        arr.append(row['SS_Article_Text'])
        arr.append(row['SS_Article_Headline'])

        if topic in per_topic.keys():
            per_topic[topic][0] += arr[0]
            per_topic[topic][1] += arr[1]
            per_topic[topic][2] += arr[2]
            per_topic[topic][3] += arr[3]
        else:
            per_topic[topic] = arr

        size += 1

    for i in range (10):
        per_topic[i][0] /= size
        per_topic[i][1] /= size
        per_topic[i][2] /= size
        per_topic[i][3] /= size

    return per_topic

def plot(myDictionary, output_file):
    ordered_dictionary = collections.OrderedDict(sorted(myDictionary.items()))

    X = ordered_dictionary.keys()

    jca = []
    jch = []
    sat = []
    sah = []

    for i in X:
        jca.append(ordered_dictionary[i][0])
    for i in X:
        jch.append(ordered_dictionary[i][1])
    for i in X:
        sat.append(ordered_dictionary[i][2])
    for i in X:
        sah.append(ordered_dictionary[i][3])

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.4, jca, 0.2, label = 'JCA')
    plt.bar(X_axis - 0.2, jch, 0.2, label = 'JCH')
    plt.bar(X_axis, sat, 0.2, label = 'SSA')
    plt.bar(X_axis + 0.2, sah, 0.2, label = 'SSH')

    plt.xticks(X_axis, X)
    plt.xlabel("Topics")
    plt.ylabel("Scores")
    plt.title("Per Comment Engagement")
    plt.legend()
    plt.savefig(output_file)
    
def main():
    data1 = pd.read_csv("../metrics_data/semantic_sim.tsv", sep='\t', low_memory=False)
    data2 = pd.read_csv("../metrics_data/data_topics.tsv", sep='\t', low_memory=False)

    pt = per_topic(data1, data2)
    plot(pt, "per_topic_difference.jpg")

if __name__ == '__main__':
    main()
