from reddit_instances import reddit
import pandas as pd
from tqdm import tqdm
import datetime
import os
from visualization import plot

def get_date(submission_id, body, author):
    submission = reddit.submission(id=submission_id)
    submission.comments.replace_more(limit=None)
    happend = False
    for comment in submission.comments.list():
        if comment.body == body:
            comment_time = comment.created
            happend = True
            break
    
    if happend == False:
        print(submission_id)
        print(body)
        print(author)
        return -1
    else:
        submission_time = submission.created
        return ((datetime.datetime.fromtimestamp(comment_time) - datetime.datetime.fromtimestamp(submission_time)))

def add_date_col(data):
    num_rows = data.shape[0]
    time_engagement = {}

    for index, row in tqdm(data.iterrows(), total=num_rows):
        submission_id = row['post id']
        body = row['comment body']
        author = row['author']
        
        date = get_date(submission_id, body, author)
        
        if date != -1:
            arr = []
            arr.append(row['Jaccard_Coef_Article'])
            arr.append(row['Jaccard_Coef_Headline'])
            arr.append(row['SS_Article_Text'])
            arr.append(row['SS_Article_Headline'])

            time_engagement[date.total_seconds()] = arr
        
    return time_engagement

def main():
    data = pd.read_csv("../metrics_data/semantic_sim.tsv", sep='\t', low_memory=False)
    time_engagement = add_date_col(data)
    plot(time_engagement, "per_time_difference.jpg", "Time Difference", "Score", "Per-Time Difference Engagement")

if __name__ == '__main__':
    main()
