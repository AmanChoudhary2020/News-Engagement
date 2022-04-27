import pandas as pd
from tqdm import tqdm
from visualization import plot

def generate_upvote_dict(data1, data2):
    num_rows = data1.shape[0]
    upvotes = {}
    
    num_comments = {}
    for index, row in tqdm(data1.iterrows(), total=num_rows):
        num_comments[row['post id']] = row['num_comments']

    num_rows = data2.shape[0]

    JCA_avg = 0
    JCH_avg = 0
    SAT_avg = 0
    SAH_avg = 0
    count = 0
    prev_id = data2.iloc[0]['post id']

    for index, row in tqdm(data2.iterrows(), total=num_rows):
        if(prev_id != row['post id']):
            arr = []
            arr.append(JCA_avg/count)
            arr.append(JCH_avg/count)
            arr.append(SAT_avg/count)
            arr.append(SAH_avg/count)
            upvotes[(row['upvotes']/num_comments[prev_id])] = arr

            prev_id = row['post id']
            
            JCA_avg = row['Jaccard_Coef_Article']
            JCH_avg = row['Jaccard_Coef_Headline']
            SAT_avg = row['SS_Article_Text']
            SAH_avg = row['SS_Article_Headline']
            count = 1
        else:
            JCA_avg += row['Jaccard_Coef_Article']
            JCH_avg += row['Jaccard_Coef_Headline']
            SAT_avg += row['SS_Article_Text']
            SAH_avg += row['SS_Article_Headline']
            count += 1

    return upvotes

def main():
    data1 = pd.read_csv("../processed_data/clean_articles.tsv", sep='\t', low_memory=False)
    data2 = pd.read_csv("../metrics_data/semantic_sim.tsv", sep='\t', low_memory=False)
    upvotes = generate_upvote_dict(data1,data2)
    plot(upvotes, "upvotes.jpg")

if __name__ == '__main__':
    main()
