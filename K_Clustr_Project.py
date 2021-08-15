#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('College_Data',index_col=0)


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[46]:


sns.lmplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private',fit_reg=False)


# In[47]:


sns.lmplot(x='Outstate',y='F.Undergrad',data=df,hue='Private',fit_reg=False)


# In[50]:


g = sns.FacetGrid(data=df,hue='Private',palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.5)


# In[51]:


g = sns.FacetGrid(data=df,hue='Private',palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.5)


# In[29]:


df.loc[df['Grad.Rate']>100,'Grad.Rate'] = 100


# In[52]:


g = sns.FacetGrid(data=df,hue='Private',palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.5)


# In[31]:


from sklearn.cluster import KMeans


# In[32]:


km = KMeans(n_clusters=2)


# In[37]:


km.fit(df.drop('Private',axis=1))


# In[38]:


km.cluster_centers_


# In[40]:


df['Cluster'] = df['Private'].apply(lambda x:1 if x=='Yes' else 0)


# In[41]:


df.head()


# In[42]:


from sklearn.metrics import classification_report,confusion_matrix


# In[43]:


print(classification_report(df['Cluster'],km.labels_))
print('\n')
print(confusion_matrix(df['Cluster'],km.labels_))


# In[ ]:




