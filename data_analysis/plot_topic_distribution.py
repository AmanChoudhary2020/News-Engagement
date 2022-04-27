import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_topic_distribution(input_file, output_file):
    data_topics = pd.read_csv(input_file, sep='\t', low_memory=False)
    myDictionary = {} 
    myDictionary = defaultdict(lambda:0,myDictionary)

    doc_topic = data_topics["doc_topic"]

    for i in doc_topic:
        myDictionary[i] += 1

    plt.xticks(np.arange(min(myDictionary.keys()), max(myDictionary.keys())+1, 1.0))

    plt.xlabel("Topic")
    plt.ylabel("Count")
    plt.title("The Guardian Articles Topic Distribution")
        
    data_topics["doc_topic"] = pd.to_numeric(data_topics["doc_topic"], errors='coerce')

    plt.bar(myDictionary.keys(), myDictionary.values())
    
    for idx, value in enumerate(myDictionary.values()):
        plt.text(value, idx, str(value))
    
    plt.savefig(output_file)

def main():
    plot_topic_distribution('data_topics.tsv', 'topic_distribution.png')

if __name__ == '__main__':
    main()
