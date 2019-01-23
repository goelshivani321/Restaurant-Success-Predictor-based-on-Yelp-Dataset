
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")


# In[32]:


comment = """RUNNING - Full Feature Engineering.
This script will run the entire feature engineering steps 
that I have implemented including both location-specific 
and non-location-specific features and generate the following files - 
1. Plots for various analyses (png)
2. Final feature engineered data files (csv & npz)
NOTE: This script needs the output files of 1.0-sm-initial-eda-and-cleaning 
AND it will fail for the downsampled dataset.
"""
print(comment)
from time import sleep
sleep(5)


# # Obtain Data

# In[3]:


data = pd.read_csv('1.0-sm-business_cleaned-1.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# # Feature Engineering

# ## Cuisine Feature

# Identifying the following popular cuisines of food that we'll use for our analysis - 
# - Mexican
# - American (new or traditional)
# - Italian
# - Indian
# - Chinese
# - Mediterranean

# In[6]:


cuisines = ['mexican', 'american', 'italian', 'indian', 
            'chinese', 'mediterranean', 'pizza', 'bar']


# In[7]:


def get_cuisine(x):
    for cuisine in cuisines:
        if cuisine in x:
            return cuisine
    else:
        return 'other'


# In[8]:


data['cuisine'] = data.categories.apply(lambda x: get_cuisine(x))


# ## Clustering Lat/Lon to identify closeby businesses

# Using DBSCAN to cluster businesses within a 2km radius i.e. eps=2kms and parameterizing such that every point gets a cluster, no noise i.e. min_samples=1

# In[9]:


coords = data[['latitude', 'longitude']].values


# In[10]:


from sklearn.cluster import DBSCAN


# In[11]:


kms_per_radian = 6371.0088
epsilon = 2 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')


# In[12]:


labels = db.fit_predict(np.radians(coords))


# In[13]:


len(set(db.labels_))


# In[14]:


data['location_cluster'] = labels


# In[15]:


data.head()


# In[17]:


sns_plot = sns.scatterplot(x = "longitude", y = "latitude", hue="location_cluster", data=data)
sns_plot.figure.savefig('clustered-lat-long.png')


# In[68]:


# import plotly
# plotly.offline.init_notebook_mode(connected=True)

# import plotly.offline as py
# data['text'] = data['name'] + '' + data['city'] + ', ' + data['state']

# scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
#     [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

# map_data = [ dict(
#         type = 'scattergeo',
#         locationmode = 'USA-states',
#         lon = data['longitude'],
#         lat = data['latitude'],
#         text = data['text'],
#         mode = 'markers',
#         marker = dict(
#             size = 8,
#             opacity = 0.8,
#             reversescale = True,
#             autocolorscale = False,
#             symbol = 'square',
#             line = dict(
#                 width=1,
#                 color='rgba(102, 102, 102)'
#             ),
#             colorscale = scl,
#             cmin = 0,
#             color = data['location_cluster'],
#             cmax = data['location_cluster'].max(),
#             colorbar=dict(
#                 title="Restaurants on Yelp"
#             )
#         ))]

# layout = dict(
#         title = 'Restaurants on Yelp <br>(Hover for business names)',
#         colorbar = True,
#         geo = dict(
#             scope='usa',
#             projection=dict( type='albers usa' ),
#             showland = True,
#             landcolor = "rgb(250, 250, 250)",
#             subunitcolor = "rgb(217, 217, 217)",
#             countrycolor = "rgb(217, 217, 217)",
#             countrywidth = 0.5,
#             subunitwidth = 0.5
#         ),
#     )

# fig = dict( data=map_data, layout=layout )
# py.iplot( fig, validate=False)


# ## Count of nearby restaurants with similar categories

# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


vect = TfidfVectorizer(analyzer='word', stop_words='english')


# In[20]:


from sklearn.metrics.pairwise import linear_kernel


# In[21]:


def count_similar(a):
    sims = np.where(a>0.9)
    return len(sims[0])


# In[22]:


def get_sim_counts(temp):
    X = vect.fit_transform(temp.categories)
    cosine_sim = linear_kernel(X)
    sim_counts = np.apply_along_axis(count_similar, 1, cosine_sim)
    return sim_counts


# In[23]:


for cluster in data.location_cluster.unique():
    temp = data[data.location_cluster == cluster]
    sim_counts = get_sim_counts(temp)
    data.loc[data.business_id.isin(temp.business_id), 'sim_counts'] = sim_counts


# In[24]:


data.sim_counts.describe()


# In[26]:


sns_plot = sns.boxplot(x='stars', y='sim_counts', data=data)
sns_plot.figure.savefig('sim-categories-stars-boxplot.png')


# ## Count of nearby restaurants with similar users visiting

# In[27]:


from sklearn.preprocessing import normalize


# In[28]:


def get_user_sim_counts(temp):
    X = normalize(temp[['mean_user_review_counts', 'mean_months_since_yelping', 'mean_user_fans', 'mean_total_compliments']])
    cosine_sim = linear_kernel(X)
    user_sim_counts = np.apply_along_axis(count_similar, 1, cosine_sim)
    return user_sim_counts


# In[29]:


for cluster in data.location_cluster.unique():
    temp = data[data.location_cluster == cluster]
    user_sim_counts = get_user_sim_counts(temp)
    data.loc[data.business_id.isin(temp.business_id), 'user_sim_counts'] = user_sim_counts


# In[30]:


data.user_sim_counts.describe()


# In[31]:


sns_plot = sns.boxplot(x='stars', y='user_sim_counts', data=data)
sns_plot.figure.savefig('sim-users-stars-boxplot.png')


# ## Exploding categories into separate columns

# In[29]:


explode = pd.DataFrame(data.categories.str.split(',').tolist(), index = data.business_id).stack()


# In[30]:


explode = explode.reset_index()[[0, 'business_id']]


# In[31]:


explode.columns = ['category', 'business_id']


# In[32]:


explode.head()


# In[33]:


explode['present'] = 1


# In[34]:


explode = explode.groupby(['business_id','category'])['present'].mean().unstack(fill_value=0)


# In[35]:


explode.head()


# In[36]:


explode.columns.name = None


# In[37]:


explode = explode.reset_index()


# In[38]:


explode.head()


# In[39]:


data = pd.merge(data, explode, left_on='business_id', right_on='business_id', how='left')


# In[40]:


data = data.drop('categories', axis=1)


# In[41]:


data.head()


# In[42]:


data.info()


# ## Transforming Categorical to Dummies

# In[43]:


data = pd.get_dummies(data, columns=['neighborhood', 'city', 'state', 'postal_code', 'cuisine'])


# In[44]:


data.head()


# ## Bucketizing review counts

# Bucketizing review counts into 10 quantile buckets

# In[45]:


data.review_count.describe()


# In[46]:


data['review_count_bucket'] = pd.qcut(data.review_count, 10)


# In[47]:


data.head()


# In[48]:


data = pd.get_dummies(data, columns=['review_count_bucket'])


# In[49]:


data.head()


# # Exporting transformed data for Modelling

# In[50]:


data.info()


# In[51]:


data.stars = (data.stars * 2).astype(np.int)


# In[52]:


data.stars.value_counts()


# In[53]:


data.info()


# In[54]:


data.select_dtypes(include='object').head()


# In[55]:


from scipy.sparse import csr_matrix, save_npz


# In[56]:


csr = csr_matrix(data.drop(['name','business_id','stars'], axis=1).values)


# In[57]:


save_npz(matrix=csr, file='1.1-sm-feature-engineered-1.npz')


# In[58]:


data.stars.to_csv('1.1-sm-feature-engineered-1-labels.csv', index=False, header=True)


# In[59]:


data.drop(['name','business_id','stars'], axis=1).to_csv('1.1-sm-feature-engineered-1.csv', index=False)

