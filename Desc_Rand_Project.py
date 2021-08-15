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


loans = pd.read_csv('loan_data.csv')


# In[4]:


loans.info()


# In[5]:


loans.head()


# In[7]:


loans.describe()


# In[8]:


sns.set_style('whitegrid')


# In[20]:


plt.figure(figsize=(20,8))
sns.displot(data=loans,x='fico',hue='credit.policy',bins=50)


# In[88]:


plt.figure(figsize=(20,8))

loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='b',label='Credit Policy = 1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='r',label='Credit Policy = 0',alpha=0.6)
plt.xlabel('FICO')
plt.legend()


# In[89]:


plt.figure(figsize=(20,8))

loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,color='b',label='Not Fully Paid= 1',alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,color='r',label='Not Fully Paid = 0',alpha=0.6)
plt.xlabel('FICO')
plt.legend()


# In[38]:


plt.figure(figsize=(12,9))
sns.displot(data=loans,x='fico',hue='not.fully.paid')


# In[90]:


plt.figure(figsize=(20,8))
sns.countplot(data=loans,x='purpose',hue='not.fully.paid',palette='Set1')


# In[93]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# In[91]:


sns.lmplot(x='fico',y='int.rate',data=loans,col='not.fully.paid',hue='credit.policy',palette='Set1')
plt.legend()


# In[50]:


loans.info()


# In[57]:


cat_feats = ['purpose']


# In[58]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[56]:


final_data.head()


# In[59]:


from sklearn.model_selection import train_test_split


# In[71]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[72]:


from sklearn.tree import DecisionTreeClassifier


# In[73]:


dtree = DecisionTreeClassifier()


# In[74]:


dtree.fit(X_train,y_train)


# In[75]:


pred_1 = dtree.predict(X_test)


# In[76]:


from sklearn.metrics import classification_report,confusion_matrix


# In[77]:


print(classification_report(y_test,pred_1))
print('\n')
print(confusion_matrix(y_test,pred_1))


# In[78]:


from sklearn.ensemble import RandomForestClassifier


# In[94]:


rand_forest = RandomForestClassifier(n_estimators=300)


# In[95]:


rand_forest.fit(X_train,y_train)


# In[96]:


pred_2 = rand_forest.predict(X_test)


# In[97]:


print(classification_report(y_test,pred_2))
print('\n')
print(confusion_matrix(y_test,pred_2))


# In[ ]:




