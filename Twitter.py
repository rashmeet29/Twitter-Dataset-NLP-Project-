import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk

from nltk.corpus import stopwords    
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import re

dataset=pd.read_csv('twitter.csv')

dataset['tweet'][0] #just to show the data

#nltk.download('stopwords')

processed_tweet=[]

for i in range(31962):
    tweet=re.sub('@[\w]*','',dataset['tweet'][i])
    tweet=re.sub('[^a-zA-Z#]','',tweet)
    tweet=tweet.lower()
    tweet=tweet.split()   #string to list
    #temp=[token for token in tweet if not token in set(stopwords.words('english'))]
    tweet=[ps.stem(token) for token in tweet if not token in set(stopwords.words('english'))]
    tweet=''.join(tweet)  #list to string
    processed_tweet.append(tweet)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)

X=cv.fit_transform(processed_tweet)
X=X.toarray()

y=dataset['label'].values

print(cv.get_feature_names())


from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()

gb.fit(X,y)
gb.score(X,y)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()

dt.fit(X,y)
dt.score(X,y)