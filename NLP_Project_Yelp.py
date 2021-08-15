#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


yelp = pd.read_csv('yelp.csv')


# In[5]:


yelp.info()


# In[6]:


yelp.head()


# In[7]:


yelp.describe()


# In[9]:


yelp['text length'] = yelp['text'].apply(len)


# In[23]:


g = sns.FacetGrid(yelp,col='stars')
g = g.map(plt.hist,'text length')


# In[24]:


sns.boxplot(yelp['stars'],yelp['text length'])


# In[25]:


sns.countplot(yelp['stars'])


# In[26]:


grouped_stars = yelp.groupby('stars').mean()


# In[27]:


grouped_stars.head()


# In[28]:


grouped_stars_corr = grouped_stars.corr()


# In[29]:


grouped_stars_corr.head()


# In[31]:


sns.heatmap(grouped_stars_corr,cmap='coolwarm',annot=True)


# In[46]:


yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]


# In[48]:


yelp_class.info()


# In[50]:


X = yelp_class['text']
y = yelp_class['stars']


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer


# In[53]:


CV = CountVectorizer()


# In[54]:


X = CV.fit_transform(X)


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[57]:


from sklearn.naive_bayes import MultinomialNB


# In[58]:


nb = MultinomialNB()


# In[59]:


nb.fit(X_train,y_train)


# In[60]:


pred = nb.predict(X_test)


# In[61]:


from sklearn.metrics import classification_report,confusion_matrix


# In[62]:


print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[63]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[64]:


from sklearn.pipeline import Pipeline


# In[66]:


pipeline = Pipeline([
    ('bow',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])


# In[69]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[70]:


pipeline.fit(X_train,y_train)


# In[72]:


pred_2 = pipeline.predict(X_test)


# In[73]:


print(confusion_matrix(y_test,pred_2))
print('\n')
print(classification_report(y_test,pred_2))


# In[75]:


pipeline = Pipeline([
    ('bow',CountVectorizer()),
    
    ('classifier',MultinomialNB())
])


# In[76]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[77]:


pipeline.fit(X_train,y_train)


# In[78]:


pred_3 = pipeline.predict(X_test)


# In[79]:


print(confusion_matrix(y_test,pred_3))
print('\n')
print(classification_report(y_test,pred_3))


# In[ ]:




