#!/usr/bin/env python
# coding: utf-8

# # WINE prediction, price and the rating of tasters.

# ### Reading the file.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

wino = pd.read_csv('winemag-data-2017-2020.csv')
wino

# Country : that the wine is from
# Points : WineEnthusiast rated the wine on a scale of 1-100
# Designation : The vineyard within the winery where the grapes that made the wine are from


# ### Understand the data

# In[2]:


wino.info()


# ### Dealing with nan values.

# In[3]:


wino.isnull().sum()/len(wino)


# In[4]:


wino['country'].value_counts(dropna=False)


# In[5]:


wino[wino['country'].isnull()]


# In[6]:


# There are only 5 rows which are nan values in 'country' column. So i decided to find out where are these wines from. 

#1
wino_country = wino['country']
wino_country[30175]
wino_country[30175] = 'South Africa'
#2
wino_country = wino['country']
wino_country[35337]
wino_country[35337] = 'Georgia'
#3
wino_country = wino['country']
wino_country[48764]
wino_country[48764] = 'South Africa'
#4
wino_country = wino['country']
wino_country[56208]
wino_country[56208] = 'South Africa'
#5
wino_country = wino['country']
wino_country[74904]
wino_country[74904] = 'Chile'


# In[7]:


wino[wino['designation'].isnull()]


# In[8]:


wino['designation'].fillna('unknown',inplace=True)


# In[9]:


wino['region_1'].fillna('unknown',inplace=True)


# In[10]:


wino['province'].fillna('unknown',inplace=True)


# In[11]:


# i dropped these rows.. i dont want to add mean value or etc.

print(wino['price'].isnull().sum())


# In[12]:


wino = wino.dropna(subset=['price'])


# In[13]:


wino = wino.reset_index(drop=True)


# In[14]:


# for my model, i dont need these columns.

wino = wino.drop(['taster_name', 'taster_photo', 'taster_twitter_handle', 'region_2'], axis=1)
wino


# In[15]:


wino.isnull().sum()


# In[16]:


wino['description'].value_counts


# ### How many unique values ?

# In[17]:


wino.winery.nunique()


# In[18]:


wino.country.value_counts()


# ### EDA. 

# In[19]:


large_countries = wino.country.value_counts()[:12]


# In[20]:


plt.figure(figsize=(10,4))

country = wino.country.value_counts()[:12]

graph = sns.countplot(x='country', 
                  data=wino[wino.country.isin(large_countries.index.values)],
                 color='olive')
graph.set_title("Countries with the largest export volume", fontsize=20)
graph.set_xlabel("Country", fontsize=15)
graph.set_ylabel("Volume", fontsize=15)
graph.set_xticklabels(graph.get_xticklabels(),rotation=45)

plt.show()


# In[21]:


plt.figure(figsize=(10, 4))
graph = sns.countplot(x='points', data=wino, color='mediumpurple')
graph.set_title("Rating Count distribuition ", fontsize=20)
graph.set_xlabel("Rating", fontsize=15) 
graph.set_ylabel("Count", fontsize=15)
plt.show()


# In[22]:


plt.figure(figsize=(16,6))

graph = sns.boxplot(x='country', y='points',
                 data=wino[wino.country.isin(large_countries.index.values)],
                 color='mediumpurple')
graph.set_title("Rating by Country", fontsize=20)
graph.set_xlabel("Country", fontsize=15)
graph.set_ylabel("Rating", fontsize=15)
graph.set_xticklabels(graph.get_xticklabels())

plt.show()


# In[23]:


#most productive regions
most_regions = wino['region_1'].value_counts()[:100].index 
print(wino[wino['region_1'].isin(most_regions)].groupby('region_1').points.mean().sort_values(ascending=False)[:20])
#Regions with the best rating from most productive onece


# In[24]:


#most productive wineries
most_wineries = wino['winery'].value_counts()[:100].index
print(wino[wino['winery'].isin(most_wineries)].groupby('winery').points.mean().sort_values(ascending=False)[:20])
#wineries with the best rating from most productive onece


# In[25]:


plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
graph1 = sns.distplot(wino['price'], color='mediumpurple')
graph1.set_title("Price distribution", fontsize=20) # seting title and size of font
graph1.set_xlabel("Price ($)", fontsize=15) # seting xlabel and size of font
graph1.set_ylabel("Frequency", fontsize=15) # seting ylabel and size of font

plt.subplot(2,1,2)
graph2 = sns.distplot(np.log(wino['price']) , color='mediumpurple')
graph2.set_title("Price Log distribution", fontsize=20) # seting title and size of font
graph2.set_xlabel("Price($)", fontsize=15) # seting xlabel and size of font
graph2.set_ylabel("Frequency", fontsize=15) # seting ylabel and size of font

graph2.set_xticklabels(np.exp(graph2.get_xticks()).astype(int))

plt.subplots_adjust(hspace = 0.3,top = 0.9)
plt.show()


# In[26]:


wino.vintage.value_counts()


# In[27]:


# Non vintage ones are '2030' in order to understand better.

i = 0
for col in wino[['vintage']]:
    for i in range(len(wino[col])):
        if wino[col][i] == "NV":
            wino[col][i] = 2030
        elif wino[col][i] == 2067:
            wino[col][i] = 2030
            
wino['vintage'] = wino['vintage'].astype(int)


# In[28]:


plt.figure(figsize=(16,20))

plt.subplot(3,1,1)

graph = sns.boxplot(x='vintage', y=np.log(wino['price']),
                    data=wino,
                    color='coral')

graph.set_title("Price by Year", fontsize=20)
graph.set_xlabel("Year", fontsize=15)
graph.set_ylabel("Price($)", fontsize=15)
graph.set_xticklabels(graph.get_xticklabels(),rotation=45)
graph.set_yticklabels(np.exp(graph.get_yticks()).astype(int))


# In[29]:


most_vintage = wino['vintage'].value_counts()[:100].index 
print(wino[wino['vintage'].isin(most_vintage)].groupby('vintage').points.mean().sort_values(ascending=False)[:20])


# In[30]:


plt.figure(figsize=(16,20))
plt.subplot(3,1,3)
graph2 = sns.boxplot(x='country', y=np.log(wino['price']),
                 data=wino[wino.country.isin(large_countries.index.values)],
                 color='coral')

graph2.set_title("Price by Country", fontsize=20)
graph2.set_xlabel("Country", fontsize=15)
graph2.set_ylabel("Price($)", fontsize=15)
graph2.set_yticklabels(np.exp(graph2.get_yticks()).astype(int))

plt.show()


# In[ ]:





# In[31]:


plt.figure(figsize=(16,6))
graph = sns.boxplot(x='country', y=wino['points']/wino['price'],
                 data=wino[wino.country.isin(large_countries.index.values)],
                 color='coral')
graph.set_title("Rating/Price by Countries", fontsize=20)
graph.set_xlabel("Country", fontsize=15)
graph.set_ylabel("Rating/Price", fontsize=15)
graph.set_xticklabels(graph.get_xticklabels())

plt.show()


# In[32]:


plt.figure(figsize=(13,5))

graph = sns.regplot(x=np.log(wino['price']), y='points', 
                    data=wino, fit_reg=False, color='olive')
graph.set_title("Rating x Price Distribuition", fontsize=20)
graph.set_xlabel("Price($)", fontsize= 15)
graph.set_ylabel("Rating", fontsize= 15)
graph.set_xticklabels(np.exp(graph.get_xticks()).astype(int))

plt.show()


# ### Checking if there is any correlation between these features

# In[33]:


corrs = wino[['points','price','vintage']].corr()

fig, ax = plt.subplots(figsize=(7,5))        

sns.heatmap(corrs,annot = True,ax=ax,linewidths=.6, cmap = 'YlGnBu');


# In[34]:


wino['variety'].value_counts().head(20)


# In[35]:


# Encoding categoricals

wino_enc = wino.copy().drop(columns = ['title'])


# In[36]:


wino_cat = [col for col in wino_enc.columns if wino_enc[col].dtype == "object"]


# In[37]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in wino_cat:
    wino_enc[col] = label_encoder.fit_transform(wino_enc[col])


# In[38]:


wino_enc


# ## Modeling

# In[39]:


y = wino_enc['price']
X = wino_enc.drop(['price'], axis = 1)


# In[40]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[42]:


X_train.head()


# In[43]:


from sklearn import linear_model

lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)


# In[44]:


from sklearn.metrics import mean_squared_error, r2_score
predictions = lm.predict(X_train)
r2_score(y_train, predictions)


# In[45]:


predictions = lm.predict(X_test)
r2_score(y_test, predictions)


# In[46]:


mse = mean_squared_error(y_test, predictions)
print(mse)


# In[47]:


import math
rmse = math.sqrt(mse)
print(rmse)


# In[48]:


correlation = wino_enc.corr()
correlation['price'].sort_values(ascending = False)


# In[49]:


#wino = wino.drop(['description'], axis=1)


# In[50]:


#wino['designation'] = wino['designation'].str.replace('-', '_')


# In[51]:


#wino['title'] = wino['title'].str.replace(';', '_')
#wino['variety'] = wino['variety'].str.replace('-', '_')


# In[52]:


#wino[wino.index == 3499]


# In[53]:


#wino.to_csv('wino2.csv', sep =';' ,index=False)


# ## Adding my csv file to MySQL

# In[54]:


import pymysql
from sqlalchemy import create_engine
import getpass  # To get the password without showing the input
password = getpass.getpass()


# In[55]:


# Most expensive 10 wines.

connection_string = 'mysql+pymysql://root:' + password + '@localhost/project'
engine = create_engine(connection_string)
query = '''SELECT *
FROM wine
ORDER BY price DESC
LIMIT 10;'''

wino_10 = pd.read_sql_query(query, engine)
wino_10


# In[56]:


connection_string = 'mysql+pymysql://root:' + password + '@localhost/project'
engine = create_engine(connection_string)
query = '''SELECT *
FROM wine
WHERE points = 100
ORDER BY price DESC;'''

wino_fav = pd.read_sql_query(query, engine)
wino_fav


# In[57]:


connection_string = 'mysql+pymysql://root:' + password + '@localhost/project'
engine = create_engine(connection_string)
query = '''SELECT *
FROM wine
ORDER BY vintage ASC
LIMIT 25;'''

wino_old = pd.read_sql_query(query, engine)
wino_old


# In[58]:


# Checking if there are any dublicates.

#wino = wino.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)


# In[59]:


# My final csv file.

#wino.to_csv('final_wino.csv',index=False)

