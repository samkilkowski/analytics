#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


data_info


# In[4]:


print(data_info.loc['revol_util']['Description'])


# In[5]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[6]:


feat_info('loan_status')


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv('lending_club_loan_two.csv')


# In[9]:


df.info()


# In[10]:


df.head()


# In[11]:


df.describe().transpose()


# In[12]:


sns.set(rc={'figure.figsize':(10,6)})


# In[13]:


sns.countplot(df['loan_status'])


# In[14]:


sns.distplot(df['loan_amnt'])


# In[15]:


df.corr()


# In[16]:


sns.heatmap(df.corr(),cmap='viridis',annot=True)


# In[17]:


feat_info('installment')


# In[18]:


feat_info('loan_amnt')


# In[19]:


sns.scatterplot(df['installment'],df['loan_amnt'])


# In[20]:


sns.boxplot(df['loan_status'],df['loan_amnt'])


# In[21]:


df.groupby('loan_status')['loan_amnt'].describe() #summary statistics


# In[22]:


df['grade'].unique()


# In[23]:


df['sub_grade'].unique()


# In[24]:


sns.countplot(df['grade'],hue=df['loan_status'])


# In[25]:


subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(data=df,x='sub_grade',order=subgrade_order,palette='coolwarm')


# In[26]:


sns.countplot(data=df,x='sub_grade',order=subgrade_order,hue='loan_status')


# In[27]:



f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


# In[28]:


sns.countplot(data=df,x='sub_grade',order=df['sub_grade'].value_counts(ascending=False).iloc[25:].index,hue='loan_status')


# In[29]:


df['loan_repaid'] = pd.get_dummies(df['loan_status'],drop_first=True)

#OR

#df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[30]:


df.corr()['loan_repaid'][:-1].sort_values().plot(kind='bar')

#OR

#df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# In[31]:


len(df)


# In[32]:


df.columns


# In[33]:


df.isnull().sum()


# In[34]:


100*(df.isnull().sum()/len(df))


# In[35]:


feat_info('emp_title')


# In[36]:


feat_info('emp_length')


# In[37]:


df['emp_title'].nunique()


# In[38]:


df['emp_title'].value_counts()


# In[39]:


df = df.drop('emp_title',axis=1)


# In[40]:


sorted(df['emp_length'].dropna().unique())


# In[41]:


emp_length_order = ['< 1 year','1 year','2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years','10+ years']


# In[42]:


sns.countplot(df['emp_length'],order=emp_length_order)


# In[43]:


sns.countplot(df['emp_length'],hue=df['loan_status'],order=emp_length_order)


# In[44]:


emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']


# In[45]:


emp_len = emp_co/emp_fp


# In[46]:


emp_len


# In[47]:


emp_len.plot(kind='bar')


# In[48]:


df = df.drop('emp_length',axis=1)


# In[49]:


df.isnull().sum()


# In[50]:


df['title']


# In[51]:


df['purpose']


# In[52]:


df = df.drop('title',axis=1)


# In[53]:


feat_info('mort_acc')


# In[54]:


df['mort_acc'].value_counts()


# In[55]:


df.corr()['mort_acc'].sort_values()


# In[56]:


df.groupby('total_acc').mean()['mort_acc']


# In[57]:


mort_fill_avg = df.groupby('total_acc').mean()['mort_acc']

def fill_in_mort(total_acc,mort_acc):
    
    if np.isnan(mort_acc):
        return mort_fill_avg[total_acc]
    else:
        return mort_acc
        
    


# In[58]:


df['mort_acc'] = df.apply(lambda x:fill_in_mort(x['total_acc'],x['mort_acc']),axis=1)


# In[59]:


df.isnull().sum()


# In[60]:


df = df.dropna()


# In[61]:


df.isnull().sum()


# In[62]:


df.select_dtypes(include='object').columns


# In[63]:


df['term'][:3]


# In[64]:



df['term'] = df['term'].apply(lambda term: int(term[:3]))


# In[65]:


df.head()


# In[66]:


df = df.drop('grade',axis=1)


# In[67]:


sub_grade_df = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[68]:


df = pd.concat([df.drop('sub_grade',axis=1),sub_grade_df],axis=1)


# In[69]:



df.head()


# In[70]:


df.select_dtypes(include='object').columns


# In[71]:


ver_stat = pd.get_dummies(df['verification_status'],drop_first=True)


# In[72]:


df = pd.concat([df.drop('verification_status',axis=1),ver_stat],axis=1)


# In[73]:


init_df = pd.get_dummies(df['initial_list_status'],drop_first=True)


# In[74]:


df = pd.concat([df.drop('initial_list_status',axis=1),init_df],axis=1)


# In[75]:


app_df = pd.get_dummies(df['application_type'],drop_first=True)


# In[76]:


df = pd.concat([df.drop('application_type',axis=1),app_df],axis=1)


# In[77]:


purpose_df = pd.get_dummies(df['purpose'],drop_first=True)


# In[78]:


df = pd.concat([df.drop('purpose',axis=1),purpose_df],axis=1)


# In[79]:


df.columns


# In[80]:


df['home_ownership'].value_counts()


# In[81]:


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')


# In[82]:


home_ownership_df = pd.get_dummies(df['home_ownership'],drop_first=True)


# In[83]:


df = pd.concat([df.drop('home_ownership',axis=1),home_ownership_df],axis=1)


# In[84]:


df['address'] = df['address'].apply(lambda address:address[-5:])


# In[85]:


df['address']


# In[86]:


address_df = pd.get_dummies(df['address'],drop_first=True)


# In[87]:


df = pd.concat([df.drop('address',axis=1),address_df],axis=1)


# In[88]:


df.columns


# In[89]:


df = df.drop('issue_d',axis=1)


# In[90]:


df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])


# In[91]:


df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x:x.year)


# In[92]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x:int(x))


# In[93]:


df = df.drop('earliest_cr_line',axis=1)


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


df = df.drop('loan_status',axis=1)


# In[96]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[98]:


from sklearn.preprocessing import MinMaxScaler


# In[99]:


scaler = MinMaxScaler()


# In[100]:


X_train = scaler.fit_transform(X_train)


# In[101]:


X_test = scaler.transform(X_test)


# In[102]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[103]:


model = Sequential()

model.add(Dense(78,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(39,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')


# In[104]:


model.fit(X_train,y_train,batch_size=256,epochs=50,validation_data=(X_test,y_test))


# In[105]:


from tensorflow.keras.models import load_model


# In[106]:


model.save('full_data_project_model.h5')  


# In[107]:


loss_df = pd.DataFrame(model.history.history)


# In[108]:


loss_df.plot()


# In[109]:


pred = model.predict_classes(X_test)


# In[110]:


from sklearn.metrics import classification_report,confusion_matrix


# In[111]:


print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
      


# In[112]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[113]:


X_train.shape


# In[114]:


model.predict_classes(new_customer.values.reshape(1,78))


# In[115]:


df.iloc[random_ind]['loan_repaid']


# In[ ]:




