
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")


# In[4]:


comment = """RUNNING - Feature Engineering for location features
This script will run the feature engineering steps 
that I have implemented for location-specific features 
and generate the following files - 
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


# In[58]:


data.head()


# In[59]:


data.info()


# # Feature Engineering

# ## Clustering Lat/Lon to identify closeby businesses

# Using DBSCAN to cluster businesses within a 2km radius i.e. eps=2kms and parameterizing such that every point gets a cluster, no noise i.e. min_samples=1

# In[60]:


coords = data[['latitude', 'longitude']].values


# In[61]:


from sklearn.cluster import DBSCAN


# In[62]:


kms_per_radian = 6371.0088
epsilon = 2 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')


# In[63]:


labels = db.fit_predict(np.radians(coords))


# In[64]:


len(set(db.labels_))


# In[65]:


data['location_cluster'] = labels


# In[66]:


data.head()


# In[67]:


sns.scatterplot(x = "longitude", y = "latitude", hue="location_cluster", data=data)


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

# In[69]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[70]:


vect = TfidfVectorizer(analyzer='word', stop_words='english')


# In[71]:


from sklearn.metrics.pairwise import linear_kernel


# In[72]:


def count_similar(a):
    sims = np.where(a>0.9)
    return len(sims[0])


# In[73]:


def get_sim_counts(temp):
    X = vect.fit_transform(temp.categories)
    cosine_sim = linear_kernel(X)
    sim_counts = np.apply_along_axis(count_similar, 1, cosine_sim)
    return sim_counts


# In[74]:


for cluster in data.location_cluster.unique():
    temp = data[data.location_cluster == cluster]
    sim_counts = get_sim_counts(temp)
    data.loc[data.business_id.isin(temp.business_id), 'sim_counts'] = sim_counts


# In[75]:


data.sim_counts.describe()


# In[76]:


sns.boxplot(x='stars', y='sim_counts', data=data)


# ## Count of nearby restaurants with similar users visiting

# In[77]:


from sklearn.preprocessing import normalize


# In[78]:


def get_user_sim_counts(temp):
    X = normalize(temp[['mean_user_review_counts', 'mean_months_since_yelping', 'mean_user_fans', 'mean_total_compliments']])
    cosine_sim = linear_kernel(X)
    user_sim_counts = np.apply_along_axis(count_similar, 1, cosine_sim)
    return user_sim_counts


# In[79]:


for cluster in data.location_cluster.unique():
    temp = data[data.location_cluster == cluster]
    user_sim_counts = get_user_sim_counts(temp)
    data.loc[data.business_id.isin(temp.business_id), 'user_sim_counts'] = user_sim_counts


# In[80]:


data.user_sim_counts.describe()


# In[81]:


sns.boxplot(x='stars', y='user_sim_counts', data=data)


# ## Transforming Categorical to Dummies

# In[82]:


data = pd.get_dummies(data, columns=['neighborhood', 'city', 'state', 'postal_code'])


# In[83]:


data.head()


# # Exporting transformed data for Modelling

# In[84]:


data.head()


# In[85]:


data.info()


# In[86]:


data.stars = (data.stars * 2).astype(np.int)


# In[87]:


data.stars.value_counts()


# In[88]:


data.info()


# In[89]:


data.select_dtypes(include='object').head()


# In[90]:


from scipy.sparse import csr_matrix, save_npz


# In[91]:


csr = csr_matrix(data.drop(['name','business_id', 'categories','stars'], axis=1).values)


# In[92]:


save_npz(matrix=csr, file='1.4-sm-feature-engineered-location.npz')


# In[93]:


data.stars.to_csv('1.4-sm-feature-engineered-location-labels.csv', index=False, header=True)


# In[94]:


data.drop(['name','business_id', 'categories','stars'], axis=1).to_csv('1.4-sm-feature-engineered-location.csv', index=False)

