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

To build the scraper, we require several things:

-[Reddit's app](https://www.reddit.com/prefs/apps)
-Reddit account (optional)

We need to create a reddit application to serve as the authenticator. 

```markdown

reddit = praw.Reddit(client_id = '', #
                    client_secret = '', #
                    user_agent = '') #

```

With this, we can use the reddit app to scrape data from any designated subreddit. For example:

```
#Top posts on r/all, the all-encompassing subreddit
hot_posts = reddit.subreddit('all').hot(limit=10)
for post in hot_posts:
    print(post.title)

```
