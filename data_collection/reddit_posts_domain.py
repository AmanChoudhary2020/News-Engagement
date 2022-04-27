'''
Collect all posts referencing The Guardian articles for a particular subreddit 
(save as TSV and Excel) from 06/2020 to 09/2020.

Subreddits: news, worldnews, qualitynews, inthenews, anythinggoesnews

Reddit Post Data: {post id, article headline, article url, post author, post upvotes, 
            # of comments on post, subreddit of post, article content, article wordcount}

dependencies:
- psaw (PushShiftAPI())
- praw
- pandas
- requests

Note: 1866 posts collected in total
TSV/Excel files found in raw_data/data_clean
'''

from reddit_instances import reddit
from reddit_instances import api
import time
import requests
import pandas as pd
import datetime as dt

def reddit_posts_domain(sub_reddits, file, start_epoch, end_epoch, domain):
    title = []
    url = []
    author = []
    upvotes = []
    num_comments = []
    subreddit_arr = []
    id_arr = []
    content = []
    wordcount = []

    guardian_articles = {}

    for subreddit in sub_reddits:
        subreddit_submissions = api.search_submissions(before=end_epoch, after=start_epoch,
                                                       subreddit=subreddit,
                                                       filter=['url', 'author', 'title', 'domain', 'id'])
        for s in subreddit_submissions:
            if(domain in s.domain and reddit.submission(s.id).num_comments > 0):
                submission = reddit.submission(s.id)
                url_parse = s.url[28:]
                index = url_parse.find('?')

                if(index != -1):
                    url_parse = url_parse[:index]

                article_text = ""
                request_needed = False

                if(url_parse in guardian_articles):
                    article_text = guardian_articles[url_parse]

                else:
                    request_needed = True
                    API_ENDPOINT = 'http://content.guardianapis.com/search'
                    MY_API_KEY = '' # add key
                    my_params = {
                        'show-fields': 'all',
                        'api-key': MY_API_KEY,
                        'ids': url_parse
                    }

                    try:
                        resp = requests.get(API_ENDPOINT, my_params)
                    except:
                        print("code 429")
                        print(int(resp.headers["Retry-After"]))
                        time.sleep(int(resp.headers["Retry-After"]))
                        resp = requests.get(API_ENDPOINT, my_params)

                    # sleep using time
                    time.sleep(1.0)
                    data = resp.json()

                    if(len(data['response']) > 0 and len(data['response']['results']) > 0):
                        guardian_articles[url_parse] = data['response']['results'][0]['fields']['bodyText']
                        article_text = data['response']['results'][0]['fields']['bodyText']

                if((request_needed and len(data['response']) > 0 and len(data['response']['results']) > 0) or not request_needed):
                    content.append(
                        article_text)
                    wordcount.append(
                        len(article_text.split()))
                    id_arr.append(s.id)
                    title.append(s.title)
                    url.append(s.url)
                    author.append(s.author)
                    upvotes.append(submission.score)
                    num_comments.append(submission.num_comments)
                    subreddit_arr.append(subreddit)

    d = {'post id': id_arr, 'title': title, 'url': url, 'author': author, 'upvotes': upvotes,
         'num_comments': num_comments, 'subreddit': subreddit_arr, 'content': content, 'wordcount': wordcount}
    r_subs_dataframe = pd.DataFrame(data=d)
    r_subs_dataframe.to_csv(file, mode='a', index=False, sep='\t')
    r_subs_dataframe.to_excel('guardian_06_07.xlsx')

def main():
    start_epoch = int(dt.datetime(2021, 6, 1).timestamp())
    end_epoch = int(dt.datetime(2021, 9, 30).timestamp())

    sub_reddits = ["news", "worldnews", "qualitynews",
                   "inthenews", "anythinggoesnews"]

    reddit_posts_domain(sub_reddits, "guardian_06_07.csv",
                        start_epoch, end_epoch, "guardian")


if __name__ == "__main__":
    main()
