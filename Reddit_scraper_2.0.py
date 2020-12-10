# -*- coding: utf-8 -*-
"""
Reddit Scraping 2.0

@author: FerrellFT

https://praw.readthedocs.io/en/latest/
"""
#%% Starter

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
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)

os.chdir('C:/Users/FerrellFT/Downloads')

#%%
#Reddit API
reddit = praw.Reddit(client_id='', client_secret='', user_agent='')

#Top posts on r/all
hot_posts = reddit.subreddit('all').hot(limit=10)
for post in hot_posts:
    print(post.author)




#%% Scraping posts

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


#%% Looking at the data

print(r_all_posts.head())
print(r_all_posts.shape)

print(r_all_posts.describe())

#Our target variable is the following:
print(r_all_posts['is_self'])

'''
The feature 'is_self' is a binary variable that denotes whether a post is a 'selftext';
that is, a post that consists only of text with no external links or images 
attached. We will use this as the target of our classifier, using the other
features scraped.

The 'url' column is redundant as it is already fulfilled by the 'is_self' column,
which describes whether a post is a self-text post or has a url attached to it.
Nevertheless, it is useful for further dives into where reddit posts link to.

For the purposes of this classifier, we can ignore this variable.

Instead of text analysis, we will be using the body item - which contains the
text attached to the post, if there are any - in a different format. We will
take into consideration instead the length of the post, denoted as the new
variable body_count below.

The item 'created' is the timestamp for the submission of the post in unix time
We will need to convert this to a datetime index

The 'distinguished' column denotes whether a post is an admin-made post or 
a generic post. Currently, its type is still an object; we need to change
this to 0 & 1 binary values.


'''

selftext_pie = px.pie(r_all_posts,names='is_self',color='is_self',title='Proportion of Selftext Posts')
selftext_pie

#%% Preprocessing
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

df['distinguished'] = df['distinguished'].fillna(0)

df['distinguished'] = df['distinguished'].astype(int)


#%% Further data exploration

df['subreddit'].unique()

df_sub = df.groupby(['subreddit'])

#%% Data splits

X = df.drop(columns=['title','is_self','id','url','body','date', 'subreddit'])
y = df['is_self']



#%% Splits
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


#%%Let's try a linear regression first
lr = LinearRegression()
lr.fit(X_train,y_train)

lr.score(X_train,y_train)
lr.score(X_test,y_test)

plt.plot(lr.coef_)
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficients')
plt.ylabel('Accuracy')
plt.show()

#The linear model score seems to perform very poorly.

#%%Try a Naive Bayes Classifier 

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

print(recall_score(y_test,NB_predict,average='binary'))
print(precision_score(y_test,NB_predict,average='binary'))


#%%Logistic regression
log = LogisticRegression()
log.fit(X_train,y_train)

log.score(X_train, y_train)
log.score(X_test,y_test)

log_predict = log.predict(X_test)
log_cmat = confusion_matrix(y_test,log_predict)

log_cmat_plot = plot_confusion_matrix(log, X_test, y_test,
                                  cmap=plt.cm.Blues,
                                  normalize='all')

print(recall_score(y_test,log_predict,average='binary'))
print(precision_score(y_test,log_predict,average='binary'))

#%%Ridge regression
from sklearn.linear_model import RidgeClassifier
#We can set up a grid search here to find the optimal value of the learning
#rate, alpha

alpharange = np.arange(start=0.05,stop=1.0,step=0.05)
ridge_trainS = []
ridge_testS = []


for a in alpharange:
    ridge = RidgeClassifier(alpha=a)
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

#%% KNN Classifier


neighbors = np.arange(10)+1
knn_trainS = []
knn_testS = []

for n in neighbors:
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    knn_trainS.append(model.score(X_train, y_train))
    knn_testS.append(model.score(X_test, y_test))

plt.plot(neighbors, knn_trainS, label="Training Accuracy")
plt.plot(neighbors, knn_testS, label="Test Accuracy")
plt.title("KNN Scores")
plt.ylabel("Accuracy")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


#%% Decision Tree
from sklearn import tree

tree_clf = DecisionTreeClassifier(max_depth=5)
tree_clf.fit(X_train,y_train)
print(tree_clf.score(X_train,y_train))
print(tree_clf.score(X_test,y_test))


fig = plt.figure(figsize=(25,20))
tree_selftext = tree.plot_tree(tree_clf,
                            feature_names=X.columns,
                           filled=True)


#Feature Importance

def plot_feature_importances(model):
    n_features = X.shape[1]
    plt.figure(figsize=(20,20))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(tree_clf)
