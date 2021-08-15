#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[119]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # lat : String variable, Latitude
# # lng: String variable, Longitude
# # desc: String variable, Description of the Emergency Call
# # zip: String variable, Zipcode
# # title: String variable, Title
# # timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# # twp: String variable, Township
# # addr: String variable, Address
# # e: String variable, Dummy variable (always 1)

# In[3]:


df = pd.read_csv('911.csv')


# In[236]:


df.info()


# In[118]:


df.head()


# In[120]:


df['zip'].value_counts().head(5)


# In[122]:


df['twp'].value_counts().head(5)


# In[123]:


df['title'].nunique()


# In[125]:


df['Reason'] = df['title'].apply(lambda x:x.split(':')[0])


# In[128]:


df['Reason'].value_counts()


# In[129]:


sns.countplot(x='Reason',data=df)


# In[133]:


df.drop(columns ='Date',inplace=True)


# In[134]:


df.drop(columns ='Month',inplace=True)


# In[135]:


df.drop(columns ='Hour',inplace=True)


# In[136]:


df.drop(columns ='Day of Week',inplace=True)


# In[138]:


type(df['timeStamp'].iloc[0]) # Verify function of iloc in lecture


# In[139]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[140]:


type(df['timeStamp'].iloc[0])


# In[142]:


df['Hour'] = df['timeStamp'].apply(lambda x:x.hour)


# In[143]:


df['Month'] = df['timeStamp'].apply(lambda x:x.month)


# In[152]:


df['Day of Week'] = df['timeStamp'].apply(lambda x:x.dayofweek)


# In[154]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[156]:


df['Day of Week']=df['Day of Week'].map(dmap)


# In[157]:


df.head()


# In[161]:


sns.countplot(x='Day of Week',data=df,hue='Reason')
plt.legend(loc=10,bbox_to_anchor=(1.25,1))


# In[162]:


sns.countplot(x='Month',data=df,hue='Reason')
plt.legend(loc=10,bbox_to_anchor=(1.25,1))


# In[163]:


# Months are missing! Try plotting another way!


# In[165]:


bymonth = df.groupby('Month').count() # See if he mentions significance of aggregation in lecture


# In[166]:


bymonth.head()


# In[237]:


bymonth['lat'].plot()


# In[170]:


#reset index in bymonth so that it can be plotted (month)

bymonth = bymonth.reset_index()


# In[171]:


bymonth.head()


# In[172]:


sns.lmplot(x='Month',y='twp',data=bymonth)


# In[175]:


df['Date'] = df['timeStamp'].apply(lambda x:x.date())


# In[178]:


groupdate = df.groupby('Date').count()


# In[181]:


groupdate['twp'].plot()
plt.tight_layout()


# In[195]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('EMS')


# In[238]:





# In[197]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('Fire')


# In[216]:


pivot_df = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack() # pay close attention to solns for this!


# In[217]:


pivot_df.head()


# In[225]:


sns.heatmap(data=pivot_df,cmap='coolwarm')


# In[227]:


sns.clustermap(data=pivot_df,cmap='twilight')


# In[228]:


pivot_df2 = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()


# In[229]:


pivot_df2.head()


# In[231]:


sns.heatmap(data=pivot_df2,cmap='rainbow_r')


# In[235]:


sns.clustermap(data=pivot_df2,cmap='viridis')


# In[ ]:




