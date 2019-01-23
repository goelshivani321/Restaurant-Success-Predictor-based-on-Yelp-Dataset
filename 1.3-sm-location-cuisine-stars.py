
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")


# In[26]:


comment = """RUNNING - Location to Cuisine to Stars Analysis.
This script will run the analysis for correlation between 
Location, Cuisine and Stars and generate the following files - 
1. Plots for various analyses (png)
NOTE: This script needs the output files of 1.0-sm-feature-engineering 
AND it will fail for the downsampled dataset.
"""
print(comment)
from time import sleep
sleep(5)


# # Obtain Data

# In[2]:


data = pd.read_csv('1.0-sm-business_cleaned-1.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# # Cuisine-to-Stars correlation

# Identifying the following popular cuisines of food that we'll use for our analysis - 
# - Mexican
# - American (new or traditional)
# - Italian
# - Indian
# - Chinese
# - Mediterranean

# In[5]:


cuisines = ['mexican', 'american', 'italian', 'indian', 
            'chinese', 'mediterranean', 'pizza', 'bar']


# In[6]:


def get_cuisine(x):
    for cuisine in cuisines:
        if cuisine in x:
            return cuisine
    else:
        return None


# In[7]:


data['cuisine'] = data.categories.apply(lambda x: get_cuisine(x))


# In[8]:


data.head(10)


# In[9]:


data.cuisine.isnull().value_counts()


# Using data only with one of the above popular cuisines.

# In[10]:


cuisine_data = data[pd.notnull(data.cuisine)]


# In[11]:


cuisine_data.info()


# In[12]:


sns.set(rc={'figure.figsize':(12,8)})
sns_plot = sns.boxplot(x="cuisine", y="stars", data=cuisine_data)
sns_plot.figure.savefig('cuisine-vs-stars-boxplot.png')


# ### Some interesting finds from the plot above
# 
# 1. Bars have usually higher median star rating as compared to other cuisines. **People really like their bars!**
# - Pizza places have large spread of star ratings, which means pizza places have a lot of passionate users coming in, who provide varying degree of ratings. **Tough business to get into!**
# - Almost every cuisine gets the lowest star rating of 1.0, but thats usually an outlier. **A hopeful message for a business owner, there will always be someone who doesn't like your food, that's ok, they're an outlier!**
# - A 5.0 star rating for an Indian restaurant is an outlier. **This can be seen as an opportunity for any aspiring Indian restaurant owner to get to the top of bunch.**
# - Mediterranean cuisine restaurants generally get a high rating, minimum of 3.5 (except outliers).
# - Almost all of the above cuisines (except pizza) have generally a high rating, i.e. 25 percentile of 3.0 and median of 3.5 star ratings. **Therefore, a business owner in general can't go wrong if they pick any of these above popular cuisines.**

# # Location-to-Stars correlation

# Lets see if there are certain postal codes or neighborhoods which have higher rated restaurants.

# In[13]:


data[data.postal_code == 'not_available'].shape


# Skipping rows where postal_code is not available.

# In[14]:


postal_code_data = data[data.postal_code != 'not_available']


# In[15]:


postal_code_data.info()


# In[16]:


def q1(x):
    return x.quantile(0.25)

def q2(x):
    return x.quantile(0.75)


# In[17]:


postal_code_stats = postal_code_data.groupby('postal_code')['stars'].agg(['min', q1, 'median', 'mean', q2, 'max', 'count']).reset_index()


# In[18]:


postal_code_stats.head()


# In[19]:


postal_code_stats.nlargest(100, 'count').nlargest(5,'median').nlargest(5,'mean')


# In[20]:


postal_code_data[
    postal_code_data.postal_code.isin(
        postal_code_stats.nlargest(100, 'count').nlargest(5,'median').nlargest(5,'mean').postal_code
    )].groupby(['postal_code','city','state']).size().reset_index()


# ### Some interesting finds from the data above

# <img src="Best Zips for Restaurants.jpg" height="1000" width="1000">  
# 
# 
# As we see above, the following 5 postal codes have the highest rated restaurants in the country even though they have the fiercest competition - 
# 1. **Postal Code - 44113, Cleveland, Ohio. Median star rating - 4.0, Mean star rating - 3.94**
# - **Postal Code - 53703, Madison, Wisconsin. Median star rating - 4.0, Mean star rating - 3.80**
# - **Postal Code - 85251, Scottsdale, Arizona. Median star rating - 4.0, Mean star rating - 3.78**
# - **Postal Code - 85016, Phoenix, Arizona. Median star rating - 4.0, Mean star rating - 3.71**
# - **Postal Code - 89101, Las Vegas, Nevada. Median star rating - 4.0, Mean star rating - 3.68**
# 
# Any new restaurant owner in this area, should be willing to face some competition but also see some good star ratings.

# # Location-to-Cuisine-to-Stars correlation

# Lets now try and find out if there are certain cuisines which are especially popular over others in a particular postal code.

# In[21]:


cuisine_data.head()


# Lets get cuisine data for above calculated 5 most popular postal codes.

# In[22]:


cuisine_postal_data = cuisine_data[cuisine_data.postal_code.isin(
    postal_code_stats.nlargest(100, 'count').nlargest(5,'median').nlargest(5,'mean').postal_code)]


# In[24]:


sns.set(rc={'figure.figsize':(20,12)})
sns_plot = sns.boxplot(x="postal_code", y="stars", hue="cuisine", data=cuisine_postal_data)
sns_plot.figure.savefig('cuisine-and-zip-vs-stars-boxplot.png')


# ## Some interesting insights from the plot above

# Here is the top cuisine for each of the 5 postal codes where we see the highest rated restaurants - 
# 
# 1. In Cleveland, Ohio 44113, **Mediterranean Cuisine** is the highest rated cuisine with median star rating of 4.25.
# - In Madison, Wisconsin 53703, **Bars** are the highest rated cuisine with median star rating of 4.0.
# - In Phoenix, Arizona 85016, **Both American and Mediterannean Cuisines** are the highest rated cuisines with median star rating of 4.0.
# - In Scottsdale, Arizona 85251, **Indian Cuisine** is the highest rated cuisine with median star rating of 4.25.
# - In Las Vegas, Nevada 89101, **Bars and Italian Cuisines** are the highest rated cuisines with median star rating of 4.0.
# 
# Other things to note - 
# 1. It is probably not a good idea to open an Indian restaurant in Madison, Wisconsin 53703.
# - Chinese cuisine restaurants are not so popular in Cleveland 44113, Madison 53703 and Scottsdale, 85251.
