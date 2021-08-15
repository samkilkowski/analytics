#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


df = pd.read_csv('KNN_Project_Data')


# In[34]:


df.head()


# In[35]:


sns.pairplot(df)


# In[36]:


from sklearn.preprocessing import StandardScaler


# In[37]:


scaler = StandardScaler()


# In[38]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[39]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[40]:


df_sc = pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[41]:


df_sc.head()


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X = df_sc

y = df['TARGET CLASS']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[44]:


from sklearn.neighbors import KNeighborsClassifier


# In[45]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[46]:


knn.fit(X_train,y_train)


# In[47]:


pred = knn.predict(X_test)


# In[48]:


from sklearn.metrics import classification_report,confusion_matrix


# In[49]:


print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))


# In[50]:


error_rate = []

for i in range(1,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))


# In[51]:


plt.figure(figsize=(12,8))
plt.plot(range(1,50),error_rate,color='b',marker='o',markerfacecolor='r',markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')

plt.title('Error Rate vs K')


# In[54]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))


# In[ ]:




