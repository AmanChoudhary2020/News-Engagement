'''
For each Reddit post collected, collects top level comments and saves to TSV and Excel files

Reddit Comment Data: {post id, comment author, post upvotes, comments wordcount, 
comment body}

dependencies:
- praw
- pandas
'''

from reddit_instances import reddit
import time
import pandas as pd
from tqdm import tqdm

def collect_comments(id_arr, file):
    file1 = file + ' .csv'
    file2 = file + ' .xlsx'
    
    author = []
    upvotes = []
    id_arr_2 = []
    wordcount = []
    comment_body = []

    for id_in in tqdm(id_arr):
        try:
            submission = reddit.submission(id=id_in)
        except:
            print("submission data failed")
        else:
            try:
                submission.comments.replace_more(limit=None)
            except:
                print("comments data failed")
            else:
                for comment in submission.comments.list():
                    same_author = False
                    same_content = False

                    for ele in author:
                        if ele == comment.author:
                            same_author = True
                            break
                    
                    for ele in comment_body:
                        if ele == comment.body:
                            same_content = True
                        
                    if((comment.parent_id == comment.link_id) and not(same_author and same_content)): #checks if given comment is top level
                        author.append(comment.author)
                        upvotes.append(comment.score)
                        id_arr_2.append(id_in)
                        comment_body.append(comment.body)
                        wordcount.append(len(comment.body.split()))
        
        time.sleep(1)

    d = {'post id': id_arr_2, 'author': author, 'upvotes': upvotes, 'wordcount': wordcount, 'comment body':comment_body}
    r_subs_dataframe = pd.DataFrame(data=d)
    r_subs_dataframe.to_csv(file1, mode='a', index=False, sep='\t')
    r_subs_dataframe.to_excel(file2)

def main():
    data = pd.read_excel('../data_clean.xlsx')
    id_arr = data['post id']
    collect_comments(id_arr,'comments_2')

if __name__ == "__main__":
    main()
