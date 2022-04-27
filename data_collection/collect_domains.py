'''
Counts the number of occurrences of each news domain in a given subreddit (save to CSV file)
Used to gauge which news domains are most popular/referenced on news-related subreddits

Subreddits: news, worldnews, qualitynews, inthenews, anythinggoesnews

Note: CSV files located in subreddit_domain_counts folder
'''

from reddit_instances import api
import pandas as pd
import datetime as dt

# counts number of occurrences of each domain in a given subreddit

def collect_domains(sub_reddit, file, start_epoch, end_epoch):
    subreddit_submissions = api.search_submissions(before=end_epoch, after=start_epoch,
                                                   subreddit=sub_reddit,
                                                   filter=['domain'])

    r_subs = {}

    for s in subreddit_submissions:
        r_subs[s.domain] = r_subs.get(s.domain, 0) + 1

    r_subs_dataframe = pd.DataFrame(
        {'domain': r_subs.keys(), 'count': r_subs.values()})
    r_subs_dataframe = r_subs_dataframe.sort_values(
        by='count', ascending=False)
    r_subs_dataframe.to_csv(file, index=False)

def main():
    sub_reddits = ["news", "worldnews", "qualitynews",
                   "inthenews", "anythinggoesnews"]
    start_epoch = int(dt.datetime(2021, 6, 1).timestamp())
    end_epoch = int(dt.datetime(2021, 9, 30).timestamp())

    for sub_reddit in sub_reddits:
        file = sub_reddit + ".csv"
        collect_domains(sub_reddit,file,start_epoch,end_epoch)

if __name__ == "__main__":
    main()
