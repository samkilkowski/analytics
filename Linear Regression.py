#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


customers = pd.read_csv('Ecommerce Customers')


# In[6]:


customers.head()


# In[7]:


customers.info()


# In[8]:


customers.describe()


# In[9]:


sns.jointplot(x=customers['Time on Website'],y=customers['Yearly Amount Spent'])


# In[10]:


sns.jointplot(x=customers['Time on App'],y=customers['Yearly Amount Spent'])


# In[46]:


sns.jointplot(x=customers['Time on App'],y=customers['Length of Membership'],kind='hex')


# In[13]:


sns.pairplot(customers)


# In[16]:


sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)


# In[18]:


X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]


# In[19]:


y = customers['Yearly Amount Spent']


# In[20]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=101)


# In[23]:


from sklearn.linear_model import LinearRegression


# In[24]:


lm = LinearRegression()


# In[25]:


lm.fit(X_train,y_train)


# In[26]:


lm.coef_


# In[27]:


print(lm.intercept_)


# In[28]:


pred = lm.predict(X_test)


# In[50]:


plt.scatter(y_test,pred)
plt.xlabel('y_test')
plt.ylabel('Predicted Values')


# In[31]:


from sklearn import metrics


# In[32]:


metrics.mean_absolute_error(y_test,pred)


# In[33]:


metrics.mean_squared_error(y_test,pred)


# In[34]:


np.sqrt(metrics.mean_squared_error(y_test,pred))


# In[51]:


metrics.explained_variance_score(y_test,pred)


# In[48]:


sns.displot(y_test-pred,bins=50,kde=True)


# In[43]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[44]:


cdf


# In[ ]:




