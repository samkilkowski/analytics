#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


ad_data = pd.read_csv('advertising.csv')


# In[4]:


ad_data.info()


# In[5]:


ad_data.head()


# In[6]:


ad_data.describe()


# In[28]:


sns.set_style('whitegrid')


# In[30]:


sns.displot(ad_data['Age'],kde=True)


# In[12]:


sns.histplot(ad_data['Age'],bins=30) #Most people in their 30's and 40's


# In[35]:


sns.jointplot(x='Age',y='Area Income',data=ad_data) #No corr


# In[33]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='r') #No corr


# In[37]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,kind='scatter',color='g') #some Corr


# In[27]:


sns.pairplot(ad_data,hue='Clicked on Ad')


# In[31]:


sns.heatmap(ad_data.isnull(),yticklabels=False,cbar=False,cmap='viridis') # No missing data here


# In[38]:


from sklearn.model_selection import train_test_split


# In[49]:


X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]

y = ad_data['Clicked on Ad']


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


lr = LogisticRegression()


# In[53]:


lr.fit(X_train,y_train)


# In[54]:


prediction = lr.predict(X_test)


# In[57]:


from sklearn.metrics import classification_report,confusion_matrix


# In[58]:


print(classification_report(y_test,prediction))


# In[59]:


confusion_matrix(y_test,prediction)


# In[ ]:




