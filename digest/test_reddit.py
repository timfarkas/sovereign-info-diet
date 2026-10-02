#!/usr/bin/env python3

import os
import praw
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "tim-filter/0.1")
)

reddit.read_only = True

subreddit = reddit.subreddit("MachineLearning")

print(f"top 10 hot posts from r/MachineLearning:\n")
print("-" * 80)

for i, submission in enumerate(subreddit.hot(limit=11)):
    if i == 0:
        continue
    
    print(f"{i}. {submission.title}")
    print(f"   score: {submission.score} | comments: {submission.num_comments}")
    print(f"   url: {submission.url}")
    print("-" * 80)