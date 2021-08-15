#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris = sns.load_dataset('iris')


# In[8]:


iris.head(2)


# In[9]:


sns.pairplot(iris,hue='species')


# In[26]:


setosa = iris[iris['species']=='setosa']
sns.kdeplot(x='sepal_width',y='sepal_length',data=setosa,cmap='plasma')


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[13]:


from sklearn.svm import SVC


# In[14]:


model = SVC()


# In[15]:


model.fit(X_train,y_train)


# In[16]:


pred = model.predict(X_test)


# In[17]:


from sklearn.metrics import classification_report,confusion_matrix


# In[18]:


print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))


# In[19]:


from sklearn.model_selection import GridSearchCV


# In[20]:


param_grid = {'C':[0.1,1,100,10,1000],'gamma':[10,1,0.1,0.01,0.001]}


# In[21]:


grid = GridSearchCV(SVC(),param_grid,verbose=4)


# In[22]:


grid.fit(X_train,y_train)


# In[23]:


grid.best_params_


# In[24]:


grid_predict = grid.predict(X_test)


# In[25]:


print(classification_report(y_test,grid_predict))
print('\n')
print(confusion_matrix(y_test,grid_predict))


# In[ ]:




