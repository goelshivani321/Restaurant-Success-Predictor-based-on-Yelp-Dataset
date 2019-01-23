
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")
import os.path


# In[25]:


comment = """RUNNING - Initial EDA and Cleaning.
This script will run the initial cleaning and exploratory analysis 
and generate the following files - 
1. Plots for various analyses (png)
2. Final cleaned data files (csv)"""
print(comment)
from time import sleep
sleep(5)


# # Obtain Data

# In[26]:


fpath = 'yelp_academic_dataset_business.csv'
business = pd.read_csv(fpath)


# In[27]:


business.head()


# In[28]:


business.info()


# # Scrub and Explore

# We will explore the following features for our study - 
# 1. Stars
# - Is Open
# - State
# - City
# - Review Count
# - Name
# - Neighborhood
# - Postal Code
# - Categories
# - Latitude/Longitude
# - Address

# ## 1. stars feature

# In[29]:


business.stars.value_counts()


# In[30]:


business.stars = business.stars.astype('category')


# In[31]:


business.stars.describe()


# In[32]:


business.stars.dtype


# No null values, all values spread among 9 categories

# ## 2. is_open feature

# In[33]:


business.is_open.value_counts()


# In[34]:


business[business.is_open == 0][['name','city','state']].head()


# Validated that these business listings are closed. e.g. https://www.yelp.com/biz/cks-bbq-and-catering-henderson?osq=CK%27S+BBQ+%26+Catering
# Only working with open businesses in this study.

# In[35]:


business = business[business.is_open == 1]


# ## 3-4 state and city features

# In[36]:


business.state.value_counts()


# In[37]:


business[business.state == 'CA'][['name','city', 'state']]


# A lot of invalid data. More than 50 states. Las Vegas shown as in CA.

# ### Problem 1 - More than 50 states

# https://statetable.com/

# In[38]:


true_states = pd.read_csv('state_table.csv')


# In[39]:


true_states.head()


# In[40]:


true_states = true_states[['name','abbreviation','country','census_region','census_division']]


# In[41]:


true_states = true_states.rename(index=str, columns={"name" : "state_full", "abbreviation" : "true_state"})


# In[42]:


true_states.head()


# In[43]:


business = pd.merge(business, true_states, left_on='state', right_on='true_state', how='left')


# In[44]:


business.head()


# In[45]:


business[business.true_state.isna()][['address','city', 'state']].head(15)


# Looks like these are businesses that are probably outside North America and since there are just 590 of them, dropping them from our analysis.

# In[46]:


business = business[pd.notnull(business.true_state)]


# In[47]:


business.country.value_counts()


# For the sake of simplicity, keeping the study just to USA.

# In[48]:


business = business[business.country == 'USA']


# In[49]:


business.info()


# In[50]:


business = business.drop(['state_full','true_state','country'], axis=1)


# ### Problem 2 - Wrong City/State combination

# https://simplemaps.com/data/us-cities

# In[51]:


cities = pd.read_csv('uscitiesv1.4.csv')


# In[52]:


cities.head()


# In[53]:


cities.city = cities.city.str.lower()


# In[54]:


cities = cities.drop(['city_ascii','county_fips','lat','lng','source','incorporated','timezone','id'], axis=1)


# In[55]:


business[['city','state']].head()


# In[56]:


business.city = business.city.str.lower()


# In[57]:


business = pd.merge(business, cities, left_on=['city','state'], right_on=['city','state_id'], how='left')


# In[58]:


business.info()


# In[59]:


business[business.state_id.isna()][['city','state']].head()


# In[60]:


business[business.state == 'CA'][['city','state_id','state']]


# There are some obvious errors like marking Las Vegas in CA or marking Charlotte as in South Carolina and some typos like 'pittsburg' instead of 'pittsburgh'. Instead of going into fixing the errors, we will drop these very few rows for now. 

# In[61]:


business = business[pd.notnull(business.state_id)]


# In[62]:


business = business.drop(['state_id'], axis=1)


# In[63]:


business.info()


# In[64]:


business.city.isnull().value_counts()


# In[65]:


business.state.isnull().value_counts()


# In[66]:


business.city.value_counts()


# ## 5. review_count feature

# In[67]:


business.review_count.describe()


# In[68]:


sns_plot = sns.boxplot(x = "stars", y = "review_count", data=business)
sns_plot.figure.savefig("review-count-boxplot.png")


# In[69]:


sns_plot = sns.distplot(business.review_count)
sns_plot.figure.savefig("review-count-distplot.png")


# In[70]:


business.review_count.quantile([.75, .9, .95, .99, .999, .9999])


# In[71]:


business.review_count.skew()


# There's heavy positive skew in the review_count feature, lets see if this valid data or not.

# In[72]:


business.sort_values(by=['review_count'], ascending=False).iloc[0]


# Manually validated from yelp the top few businesses and their corresponding review counts. Looks good. Lets see if we can transform the feature to make it more gaussian and better distributed

# In[73]:


sns_plot = sns.distplot(np.log10(business.review_count))
sns_plot.figure.savefig("log-review-count-distplot.png")


# In[74]:


sns_plot = sns.boxplot(x=business.stars, y=np.log10(business.review_count))
sns_plot.figure.savefig("log-review-count-boxplot.png")


# As we see above, there are usually lesser number of reviews for very low and very high ratings. The inter-quartile range is similar for each of them which implies people generally have concensus on a business' star ratings. The more common star ratings, i.e. 3-4.5 have higher median number of review counts. 

# In[55]:


np.log10(business.review_count).skew()


# Log10 transform of review counts looks like a good valid feature to have. Keeping the transform. Making sure correlation with 'stars' remains unaffected before doing the transform.

# In[56]:


business.stars.corr(business.review_count)


# In[57]:


business.stars.corr(np.log10(business.review_count))


# In[58]:


business.review_count = np.log10(business.review_count)


# In[59]:


business.review_count.isnull().value_counts()


# ## 6. name feature

# In[60]:


business.name.head()


# In[61]:


business.name.isnull().value_counts()


# In[62]:


business.name = business.name.str.lower()


# ## 7. neighborhood feature

# In[63]:


business.neighborhood.head()


# In[64]:


business[['city','neighborhood','address','name','postal_code']].head()


# In[65]:


business.neighborhood.isnull().value_counts()


# In[66]:


business.groupby(['city','neighborhood'])['business_id'].count()


# Instead of replacing neighborhoods with the most common neighborhood, I'm replacing NAs with a generic 'NOT_AVAILABLE' value.

# In[67]:


business.neighborhood = business.neighborhood.fillna('NOT_AVAILABLE')


# In[68]:


business.neighborhood = business.neighborhood.str.lower()


# In[69]:


business.neighborhood.isnull().value_counts()


# ## 8. postal_code feature

# In[70]:


business.postal_code.head()


# In[71]:


business.postal_code.isnull().value_counts()


# In[72]:


business[business.postal_code.isnull()][['name','address', 'city', 'state']].head()


# Checked on Yelp, looks like the businesses with NaN addresses actually have address missing on Yelp. e.g. https://www.yelp.com/biz/xtreme-cleaning-az-phoenix-2?osq=Xtreme+Cleaning+AZ

# In[73]:


business[pd.notnull(business.address) & business.postal_code.isnull()][['name','address', 'city', 'state']].head()


# Even the rows where address is not NaN, there are postal_code missing for these listings on Yelp. e.g. https://www.yelp.com/biz/monroe-street-farmers-market-madison?osq=Monroe+Street+Farmer%27s+Market. Replacing all NaN values with 'NOT_AVAILABLE'

# In[74]:


business.postal_code = business.postal_code.fillna('NOT_AVAILABLE')


# In[75]:


business.postal_code = business.postal_code.str.lower()


# In[76]:


business.postal_code.isnull().value_counts()


# ## 9. categories feature

# In[77]:


business.categories.head()


# In[78]:


business.categories.isnull().value_counts()


# In[79]:


business[business.categories.isnull()][['name','city','state']].head()


# Dropping 392 rows with Null categories.

# In[80]:


business = business[pd.notnull(business.categories)]


# In[81]:


business.categories = business.categories.str.lower()


# In[82]:


business.categories.str.contains('restaurant|food').value_counts()


# Keeping only food/restaurant related listings for our study.

# In[83]:


business = business[business.categories.str.contains('restaurant|food')]


# ## 10. latitude/longitude feature

# In[84]:


business.latitude.isnull().value_counts()


# In[85]:


business.longitude.isnull().value_counts()


# In[86]:


business[business.longitude.isnull()]['latitude']


# In[87]:


business[business.latitude.isnull()]['longitude']


# In[88]:


business = business[pd.notnull(business.latitude) & pd.notnull(business.longitude)]


# In[89]:


business[['latitude','longitude']].describe()


# Got the bounding box i.e. max/min lat longs for continental United States from here - https://en.wikipedia.org/wiki/List_of_extreme_points_of_the_United_States

# In[90]:


top = 49.3457868 # north lat
left = -124.7844079 # west long
right = -66.9513812 # east long
bottom =  24.7433195 # south lat


# In[91]:


business[(business.latitude > top) | (business.latitude < bottom) | (business.longitude > right) | (business.longitude < left)][['name','city','state']]


# There are about 12 business who's latitude/longitude fall outside the bounding box of continental USA. Dropping them

# In[92]:


business = business[(business.latitude <= top) & (business.latitude >= bottom) & (business.longitude<=right) & (business.longitude>=left)]


# ## 11. Address

# In[93]:


business.address.isnull().value_counts()


# In[94]:


business[business.address.isnull()][['name','address','city','state']].head()


# As we see a lot of the places above with null address are actually mobile restaurants, like food trucks etc. Marking all these addresses as 'NOT_AVAILABLE'.

# In[95]:


business.address = business.address.fillna('NOT_AVAILABLE')


# In[96]:


business.address = business.address.str.lower()


# In[97]:


business.address.isnull().value_counts()


# # Join Data

# ## Joining with tip data

# Get all user_ids that provided tips for this business

# In[100]:


tip = pd.read_csv('yelp_academic_dataset_tip.csv')


# In[102]:


tip.head()


# In[105]:


business = business.merge(tip[['business_id','user_id']], left_on='business_id', right_on='business_id', how='left')


# In[110]:


business = business.merge(business.groupby('business_id').apply(lambda x: x['user_id'].unique()).reset_index(), 
               left_on='business_id', right_on='business_id', how='left')


# In[118]:


business = business.drop_duplicates(subset='business_id')


# In[121]:


business['users'] = business[0]


# In[124]:


business = business.drop(['user_id',0], axis=1)


# ## Joining with review data

# Get all user_ids that provided reviews for this business

# In[125]:


review = pd.read_csv('yelp_academic_dataset_review.csv')


# In[127]:


review.head()


# In[128]:


business = business.merge(review[['business_id','user_id']], left_on='business_id', right_on='business_id', how='left')


# In[131]:


business = business.merge(business.groupby('business_id').apply(lambda x: x['user_id'].unique()).reset_index(), 
               left_on='business_id', right_on='business_id', how='left')


# In[132]:


business = business.drop_duplicates(subset='business_id')


# In[135]:


business['users2'] = business[0]


# In[137]:


business = business.drop(['user_id',0], axis=1)


# In[155]:


business = business.reset_index(drop=True)


# ## Union of all users interacting with this business

# In[171]:


def union(x):
    a = x['users']
    b = x['users2']
    return list(set(a) | set(b))


# In[173]:


business.users = business.apply(union, axis=1)


# In[176]:


business = business.drop(['users2'], axis=1)


# ## Joining user profiles with business

# In[177]:


users = pd.read_csv('yelp_academic_dataset_user.csv')


# In[212]:


users.head()


# In[205]:


users.yelping_since = users.yelping_since.astype(np.datetime64)


# In[218]:


users['months_since_yelping'] = ((pd.to_datetime('today') - users.yelping_since)/np.timedelta64(1, 'M'))


# In[221]:


users.columns


# In[243]:


users['total_compliments'] = users[users.columns[users.columns.to_series().str.contains('compliment')]].sum(axis=1)


# In[244]:


def get_user_stats(x):
    data = users[users.user_id.isin(x['users'])]
    result = x[['business_id']]
    result['mean_user_review_counts'] = data.review_count.mean()
    result['mean_months_since_yelping'] = data.months_since_yelping.mean()
    result['mean_user_fans'] = data.fans.mean()
    result['mean_total_compliments'] = data.total_compliments.mean()
    return result


# In[247]:


business = business.merge(business.apply(get_user_stats, axis=1), left_on='business_id', right_on='business_id', how='left')


# In[250]:


business.info()


# # Writing out cleaned data

# In[251]:


cleaned_cols = ['stars', 'business_id', 'name', 'neighborhood',
               'city', 'state', 'postal_code', 'latitude', 'longitude',
               'categories', 'review_count', 'mean_user_review_counts', 
                'mean_months_since_yelping', 'mean_user_fans', 'mean_total_compliments']


# In[252]:


business[cleaned_cols].head()


# In[253]:


business[cleaned_cols].to_csv('1.0-sm-business_cleaned-1.csv', index=False)


# In[254]:


business.to_csv('1.0-sm-business_cleaned-raw.csv', index=False)

