#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score
import warnings
warnings.filterwarnings('ignore')


# In[11]:


Dataset = pd.read_csv('IMDB.csv', encoding = 'ISO-8859-1')
Dataset.head()


# In[13]:


Dataset.info()


# In[16]:


attribute = Dataset.columns
print(attribute)


# In[17]:


Dataset.isnull().sum()


# In[23]:


shape = Dataset.shape
print("number of rows:",{shape[0]}, "Number of columns:", {shape[1]})


# In[26]:


unique_geners = Dataset['Genre'].unique()
print("unique Generes",unique_geners)


# In[27]:


rating_dist = Dataset['Rating'].value_counts()
print("Rating Distribution:\n", rating_dist)


# In[28]:


Dataset.drop_duplicates(inplace = True)


# In[30]:


attributes = ['Name','Year','Duration','Votes','Rating']
Dataset.dropna(subset = attributes, inplace = True)
missing_value = Dataset.isna().sum()
print(missing_value)


# In[31]:


Dataset


# In[33]:


movie_name_rating = Dataset[['Name','Rating']]
print(movie_name_rating.head())


# In[51]:


top_rated_movies = Dataset.sort_values(by = 'Rating',ascending = False).head(15)
plt.figure(figsize = (10,6))
plt.barh(top_rated_movies['Name'], top_rated_movies['Rating'],color = 'yellow')
plt.xlabel('Rating')
plt.ylabel('Movie')
plt.title('Top 15 Highest-Rated Movies')
plt.gca().invert_yaxis()
plt.show()


# In[58]:


Dataset['Votes'] = pd.to_numeric(Dataset['Votes'],errors = 'coerce')
plt.figure(figsize = (14,10))
plt.scatter(Dataset['Rating'],Dataset['Votes'],alpha = 0.6,color = 'b')
plt.ylabel('Rating')
plt.ylabel('Votes')
plt.title("Scatter plot of Rating vs. Votes")
plt.grid(True)
plt.show()


# In[66]:


actors = pd.concat([Dataset['Actor 1'], Dataset['Actor 2'],Dataset['Actor 3']])
actor_counts = actors.value_counts().reset_index()
actor_counts.columns = ['Actor', 'Number of Movies']
plt.figure(figsize = (12,6))
sns.barplot(x = 'Number of Movies', y = 'Actor', data = actor_counts.head(10),palette = 'viridis')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.title("Top 10 Actors by number of Movies performd")
plt.show


# In[68]:


columns_of_intrest = ['Votes','Rating','Duration','Year']
sns.set(style = 'ticks')
sns.pairplot(Dataset[columns_of_intrest], diag_kind = 'kde', markers = 'o', palette = 'viridis',height = 2.5, aspect = 1.2)
plt.suptitle('Pair Plot of Voting, Rating,Duration, and Year', y = 1.02)
plt.show()


# In[69]:


numerical_columns = ['Votes','Rating','Duration','Year']
correlation_matrix = Dataset[numerical_columns].corr()
plt.figure(figsize = (8,6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', vmin = -1,vmax = 1)
plt.title('Correlation Heatmap')
plt.show()


# In[70]:


dataset_sorted  = Dataset.sort_values(by = 'Votes', ascending = False)
dataset_sorted['Vote_Count_Percentile'] = dataset_sorted['Votes'].rank(pct = True)*100
dataset_sorted.reset_index(drop = True, inplace = True)
print(dataset_sorted[['Name','Votes','Vote_Count_Percentile']])


# In[71]:


Dataset.head()


# In[72]:


Dataset = Dataset.dropna(subset = ['Votes'])
Dataset


# In[82]:


Dataset['Year'] = Dataset['Year'].astype(str)
Dataset['Duration'] = Dataset['Duration'].astype(str)
Dataset['Year'] = Dataset ['Year'].str.extract('(\d+)').astype(float)
Dataset['Duration'] = Dataset['Duration'].str.extract('(\d+)').astype(float)
x = Dataset[['Year','Duration','Votes']]
y = Dataset['Rating']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)


# In[83]:


model = LinearRegression()


# In[85]:


model.fit(x_train,y_train)


# In[86]:


y_pred = model.predict(x_test)


# In[88]:


mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred,squared = False)
r2 = r2_score(y_test,y_pred)
print("MEA",mae)
print("RMSE",rmse)
print("R2_ score", r2)


# In[95]:


y_test = np.random.rand(100)*10
y_pred = np.random.rand(100)*10
errors = y_test - y_pred
fig, axs = plt.subplots(3,1,figsize = (8,12))

axs[0].scatter(y_test,y_pred)
axs[0].set_xlabel("Actual Rating")
axs[0].set_ylabel("Predicted Rating")
axs[0].set_title("Actual vs. Predicted Ratings")

movie_samples = np.arange(1,len(y_pred)+1)
axs[1].plot(movie_samples, y_pred, marker = 'o', linestyle = '-')
axs[1].set_xlabel("Movie Samples")
axs[1].set_ylabel("Predicted Ratings")
axs[1].set_title("Predicted Rating Across Movie Samples")
axs[1].tick_params(axis = 'x',rotation = 50)

axs[2].hist(errors,bins=30)
axs[2].set_xlabel("Prediction Errors")
axs[2].set_ylabel("Frequency")
axs[2].set_title("Disrtibution of prediction errors")
axs[2].axvline(x = 0, color = 'r', linestyle = '--')
plt.tight_layout()
plt.show()


# In[ ]:




