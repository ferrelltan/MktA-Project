## Scraping Reddit Post Data
This small project takes a look at reddit data utilizing the Reddit API and the PRAW package. Reddit is a highly popular social media website centered around smaller clumps of communities called "subreddits" centered on a particular topic - whether it's general topics like cooking or politics or even niche communities. Inevitably, this may lead to fragmented communities that become echo chambers. Alternatively, subreddits can become heavily contested by users with a variety of opinions. Reddit traditionally has a western audience, but has begun to become increasingly global in its userbase. We can look to the reddit userbase in a similar manner to other social media websites to see where common trends are moving, whether it's technology, pop culture, political ideas or others. Organizations seeking close engagement with social media may wish to see reddit as an avenue for gauging user interest.

Through this project, we can visualize and analyze Reddit posts and comments in a similar vein to review websites and social media. The challenge lies in sifting through the myriad of comments that do not necessarily reflect user sentiment.

For this, we require the [python package _praw_](https://praw.readthedocs.io/en/latest/).

**NOTE**: With how Github pages is set up, there is no way for me to include pictures into this page. So alternatively, I have placed the visualizations in the issues tab on this repository.

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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score

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

To create the reddit app, click on the link above and click the create application option (note: this may require you to create a reddit account). Once you've created the application, the application will have several information that PRAW will require to create the scraper. 
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


### The Scraper

We can utilize reddit data to illustrate certain trends in the userbase. There is a great number of data you can pull via the PRAW package. These are separated into 3 different types: subreddits, posts and comments. For the purposes of this analysis, we will be focusing on user posts. Under these posts, there are a variety of different attributes we can retrieve; this ranges from attributes like post scores, upvote (like/dislike) ratio and others.

We can set up the scraper to scrape posts in the different 'subreddits' (message boards/forums) dedicated to specific fields. In this case, we will be using the subreddit 'all', which serves as the front page of reddit where posts with the highest metrics (i.e. comments, upvotes) or the fastest traction (i.e. high number of comments in a short time span) in a given time period appear from across all of the different subreddits. This serves as a way to gauge reddit data as a whole, though the caveat remains that some subreddits - particularly controversial or R18 ones - are typically prohibited from appearing in r/all. Furthermore, since this subreddit only accounts for posts with the highest metrics or fastest traction, this may not capture reddit as a whole. Either way, we can easily substitute the subreddit we wish to scrape from by replacing 'all' by a given subreddit.

```markdown
#Setting up the scraper
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

One thing to note however is that the dataset I obtained is heavily imbalanced towards the negative ('is_self' = False). So, we will try to tackle this by undersampling (a 60-40 train-test split).

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

### Analysis: Classifying selftext posts

Now that we have our processed dataframe, we will be able to attempt to classify selftext posts.

```
#Data Splits
X = df.drop(columns=['title','is_self','id','url','body','date', 'subreddit'])
y = df['is_self']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
```

The first thing we can try is a linear regression.

```
#Linear Regression
lr = LinearRegression()
lr.fit(X_train,y_train)

print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))

plt.plot(lr.coef_)
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficients')
plt.ylabel('Accuracy')
plt.show()
```
The linear regression seems to perform very poorly overall on both the training and test sets. We will try several different models:

#Ridge Regression
```
#We can set up a grid search here to find the optimal value of the learning
#rate, alpha

alpharange = np.arange(start=0.05,stop=1.0,step=0.05)
ridge_trainS = []
ridge_testS = []


for a in alpharange:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train,y_train)
    ridge_trainS.append(ridge.score(X_train,y_train))
    ridge_testS.append(ridge.score(X_test,y_test))

plt.plot(alpharange, ridge_trainS, label="Training Accuracy")
plt.plot(alpharange, ridge_testS, label="Test Accuracy")
plt.title("Ridge Scores")
plt.ylabel("Accuracy")
plt.xlabel("Alpha")
plt.grid()
plt.legend()
```
The ridge regression performs similarly poorly across the different learning rates.

#Naive Bayes Classifier
```
# Naive Bayes Classifier
NB = GaussianNB()
NB.fit(X_train,y_train)

print(NB.score(X_train,y_train))
print(NB.score(X_test,y_test))

#Confusion matrix

NB_predict = NB.predict(X_test)
NB_cmat = confusion_matrix(y_test,NB_predict)

NB_cmat_plot = plot_confusion_matrix(NB, X_test, y_test,
                                  cmap=plt.cm.Blues,
                                  normalize='true')

print(recall_score(y_test,NB_predict,average=None))
print(precision_score(y_test,NB_predict,average=None))
```
The Naive Bayes Classifier seems to perform incredibly well, with a test score of around 0.95 despite undersampling. Looking at the confusion matrix, however, we see that both the recall and precision is still relatively low. 

#Logistic Regression
```

log = LogisticRegression()
log.fit(X_train,y_train)

log.score(X_train, y_train)
log.score(X_test,y_test)

log_predict = log.predict(X_test)
log_cmat = confusion_matrix(y_test,log_predict)

log_cmat_plot = plot_confusion_matrix(log, X_test, y_test,
                                  cmap=plt.cm.Blues,
                                  normalize='all')
```
The logistic regression seems to score well in terms of model accuracy. But the confusion matrix indicates that the model prediction is very inaccurate.

```

```
