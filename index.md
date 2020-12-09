## Scraping Reddit Data

This small project takes a look at reddit data utilizing the Reddit API and the PRAW package. Reddit is a highly popular social media website centered around smaller clumps of communities called "subreddits" centered on a particular topic - whether it's general topics like cooking or politics or even niche communities. Inevitably, this may lead to fragmented communities that become echo chambers, like r/conservative which is frequented by conservative-minded people. Alternatively, subreddits can become heavily contested by users with a variety of opinions. Reddit traditionally has a western audience, but has begun to become increasingly global in its userbase. 

Through this project, we can visualize and analyze Reddit posts and comments in a similar vein to review websites and social media. The challenge lies in sifting through the myriad of comments that do not necessarily reflect user sentiment.

For this, we require the [python package _praw_](https://praw.readthedocs.io/en/latest/).


### Starter
```markdown
# Packages and settings

import praw
import os
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes  import GaussianNB
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)
'''

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

### Building the scraper

To build the scraper, we require:

[Reddit's app](https://www.reddit.com/prefs/apps)

We need to create a reddit application to serve as the authenticator. This is done via the PRAW package, which connects to the application you can create from the link above. The application is needed to serve as an authenticator to allow Reddit scraping; it is possible to do this via a different package like BeautifulSoup, but the PRAW package is heavily personalized to tackle reddit scraping.

To create the reddit app, click on the link above and click the create application option (note: this may require you to create a reddit account). Once you've created the application, the application will have several information that praw will require to create the scraper:

```markdown

reddit = praw.Reddit(client_id = '', #Id name under your application name
                    client_secret = '', #The 'secret' password value
                    user_agent = '') #The name of the application

```

With this, we can use the reddit app to scrape data from any designated subreddit. For example:

```
#Top posts on r/all, the all-encompassing subreddit
hot_posts = reddit.subreddit('all').hot(limit=10)
for post in hot_posts:
    print(post.title)

```


## Data Analysis

We can utilize reddit data to illustrate certain trends in the userbase. There is a great number of data you can pull via the PRAW package. These are separated into 3 different types: subreddits, posts and comments. For the purposes of this analysis, we will be focusing on user posts.

```markdown
posts = []

r_all = reddit.subreddit('all')

for post in r_all.top(limit=750):
    posts.append([post.title, 
                  post.score, 
                  post.id, 
                  post.subreddit,
                  post.is_original_content,
                  post.distinguished,
                  post.over_18,
                  post.spoiler,
                  post.is_self,
                  post.upvote_ratio,
                  post.url, 
                  post.num_comments, 
                  post.selftext,
                  post.created])

r_all_posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit','original_content', 
                                          'distinguished', 'R18', 'spoiler', 'is_self', 'upvote_ratio', 
                                          'url', 'num_comments', 'body', 
                                          'created'])
```
