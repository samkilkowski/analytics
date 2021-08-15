#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 
import chart_studio.plotly as py


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('2014_World_Power_Consumption')


# In[5]:


df.head()


# In[26]:


data = dict(type = 'choropleth',
            colorscale = 'viridis',
            reversescale=True,
            locations=df['Country'],
            locationmode='country names',
            z=df['Power Consumption KWH'],
            text=df['Text'],
            colorbar={'title':"Each Country's power usage in KWH"})


# In[27]:


layout = dict(title ='Country KWH usage',geo = dict(showframe=False,projection={'type':'mercator'}))


# In[28]:


chromap = go.Figure(data=[data],layout=layout)


# In[29]:


iplot(chromap)


# In[30]:


df2 = pd.read_csv('2012_Election_Data')


# In[31]:


df2.head()


# In[39]:


data = dict(type = 'choropleth',
            colorscale = 'inferno',
            reversescale=True,
            locations=df2['State Abv'],
            locationmode='USA-states',
            z=df2['Voting-Age Population (VAP)'],
            text=df2['State'],
            colorbar={'title':"Voting-Age Population (VAP)"})


# In[40]:


layout = dict(title = 'Each state VAP',geo = dict(scope='usa',showlakes=True,lakecolor='rgb(85,173,240)'))


# In[41]:


chromap2 = go.Figure(data=[data],layout=layout)


# In[42]:


iplot(chromap2)


# In[ ]:




