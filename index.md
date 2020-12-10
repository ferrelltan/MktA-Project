## Reddit Scraper

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

## Data Exploration

```markdown
print(r_all_posts.head())
print(r_all_posts.shape)

print(r_all_posts.describe())

print(r_all_posts['is_self'])

selftext_pie = px.pie(r_all_posts,names='is_self',color='is_self',title='Proportion of Selftext Posts')
selftext_pie

```
In this analysis, we want to determine how accurate we can predict whether a given reddit post is a selftext post. To do so, we will need to inspect the feature itself. This target feature, 'is_self' is a binary variable that denotes whether a post is a 'selftext'; that is, a post that consists only of text with no external links or images 
attached. We will use this as the target of our classifier, using the other features scraped.

Firstly, we will remove some features we don't require:

The 'url' column is redundant as it is already fulfilled by the 'is_self' column, which describes whether a post is a self-text post or has a url attached to it. Nevertheless, it is useful for further dives into where reddit posts link to. The 'title' feature is also removed, as it contains only the title of the post. Alternatively, we can use a word count method to capture the length of the title, but I opted not to include that since I will be doing so for the 'body' feature (see below). However, I can see how it may actually be a useful feature since post titles act in a similar manner to video or article titles - maybe via a 'clickbait' style.

Next, we will need to process the data:

As indicated before, instead of text analysis, we will be using the body item - which contains the text attached to the post, if there are any - in a different format. We will
transform the variable into a count of how many words are in the body, contained in the new variable body_count.

The item 'created' is the timestamp for the submission of the post in unix time. We will need to convert this to a datetime index

The 'distinguished' column denotes whether a post is an admin-made post or a generic post. Currently, we cannot train the model with this feature due to it being the wrong object type, but since it is a binary variable we can turn this into integers 0 & 1.

```
#Create a wordcount for text in body

def word_count(string):
    return(len(string.strip().split(" ")))

body_wc = []

for words in r_all_posts.body:
    body_wc.append(word_count(words))

body_wordcount = pd.DataFrame(body_wc,columns=['body_count'])

body_wordcount['id'] = r_all_posts['id']

df = r_all_posts.merge(body_wordcount, left_on='id', right_on='id')

df['body_count'].unique()
df.columns

#Converting from unix to datetime
from datetime import datetime
print(datetime.fromtimestamp(df['created'][1]))

df['date'] = np.zeros(750)
for i in range(750):
    df['date'][i] = datetime.fromtimestamp(df['created'][i])

type(df['date'][1])
df['date']

#df.drop(columns='',inplace=True)

#Now, we can create separate columns for more specific date/time ranges
df['year'] = pd.DatetimeIndex(df['date']).year
df['dayofweek'] = pd.DatetimeIndex(df['date']).dayofweek
df['month'] = pd.DatetimeIndex(df['date']).month


df['distinguished'].unique()
df['distinguished'] = df['distinguished'].map({'admin':1})
df['distinguished'] = df['distinguished'].fillna(0) #because the feature previously either returns 'admin' or an empty space, we need to fill the NA values with 0
df['distinguished'] = df['distinguished'].astype(int)
```

