#!/usr/bin/env python
# coding: utf-8

# In[79]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)


# In[11]:


BAC = data.DataReader('BAC','yahoo',start,end)


# In[13]:


C= data.DataReader('C','yahoo',start,end)


# In[14]:


GS = data.DataReader('GS','yahoo',start,end)


# In[15]:


JPM = data.DataReader('JPM','yahoo',start,end)


# In[16]:


MS = data.DataReader('MS','yahoo',start,end)


# In[17]:


WFC = data.DataReader('WFC','yahoo',start,end)


# In[18]:


tickers = ['BAC','C','GS','JPM','MS','WFC']


# In[22]:


bank_stocks = pd.concat([BAC,C,GS,JPM,MS,WFC],axis=1,keys=tickers)


# In[23]:


bank_stocks.columns.names = ['Bank Ticker','Stock Info']


# In[24]:


bank_stocks.head()


# In[30]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# In[31]:


returns = pd.DataFrame()


# In[41]:


for tick in tickers:
    returns[tick +' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()


# In[42]:


import seaborn as sns


# In[44]:


sns.pairplot(returns[1:])


# In[45]:


returns.idxmin()


# In[47]:


returns.idxmax()


# In[46]:


returns.std()


# In[54]:


returns.loc['2015-01-01':'2015-12-31'].std()


# In[58]:


sns.displot(data=returns.loc['2015-01-01':'2015-12-31']['MSReturns'],color='b',bins=100,kde=True)


# In[59]:


sns.displot(data=returns.loc['2008-01-01':'2008-12-31']['CReturns'],color='g',bins=100,kde=True)


# In[60]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# In[61]:


for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,8),label=tick)
plt.legend()


# In[64]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot(figsize=(12,8))
plt.tight_layout()


# In[66]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()


# In[70]:


plt.figure(figsize=(12,8))
BAC['Close'].loc['2008-01-01':'2008-12-31'].rolling(window=30).mean().plot()
BAC['Close'].loc['2008-01-01':'2008-12-31'].plot()
plt.legend()


# In[73]:


sns.heatmap(data=bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[74]:


sns.clustermap(data=bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[75]:


BAC[['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle')


# In[78]:


MS[['Open','High','Low','Close']].loc['2015-01-01':'2015-12-31'].ta_plot(study='sma',periods=[13,21,25])


# In[77]:


BAC[['Open','High','Low','Close']].loc['2015-01-01':'2015-12-31'].ta_plot(study='boll')


# In[ ]:




