'''
dependencies for Reddit data collection:
- praw
- psaw (PushshiftAPI())
'''

import praw
from psaw import PushshiftAPI

api = PushshiftAPI()

# Initiate Reddit instance
reddit = praw.Reddit(
    user_agent="",
    client_id="",
    client_secret="",
    username="",
    password="",
)
