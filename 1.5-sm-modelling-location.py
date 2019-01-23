
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")


# In[1]:


comment = """RUNNING - Modelling with Location-Specific Feature Engineered data.
This script will run the model training for location-specific feature engineered data
that I have generated in previous notebook
NOTE: This script needs the output files of 1.4-sm-feature-engineering-location 
AND it will fail for the downsampled dataset.
"""
print(comment)
from time import sleep
sleep(5)


# # Obtain Data

# In[27]:


from scipy.sparse import csr_matrix, load_npz


# In[28]:


train = load_npz('1.4-sm-feature-engineered-location.npz')
# train = pd.read_csv('1.1-am-feature-engineered-1.csv')


# In[29]:


train.shape


# In[30]:


labels = pd.read_csv('1.4-sm-feature-engineered-location-labels.csv', )


# In[31]:


labels.info()


# In[32]:


labels.stars.values


# ### Modelling label as binary

# Here I am defining the success measure of a restaurant as being a star rating of >= 3.5 (i.e. label >=7)

# In[33]:


labels = labels >= 7


# In[34]:


labels = labels.astype(np.int)


# In[35]:


labels.stars.value_counts()


# # Setup for Modelling

# ## Splitting data

# In[36]:


X = train
y = labels.stars.values


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_cv, y_train, y_cv = train_test_split( 
              X, y, test_size = 0.4, random_state = 42, stratify = y) 


# In[39]:


X_cv, X_test, y_cv, y_test = train_test_split(X_cv, y_cv, test_size=0.1, random_state=42, stratify=y_cv)


# In[40]:


print(X_train.shape)
print(X_cv.shape)
print(X_test.shape)


# In[41]:


y_train


# ## Training Function

# In[42]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[43]:


def trainModelGrid(estimator, params, train, y, cv=5):
    model = GridSearchCV(estimator, params, n_jobs=-1, scoring='accuracy',cv=cv)
    model.fit(train, y)
    return model


# In[44]:


def trainModelRandom(estimator, params, train, y, cv=5, n_iter=5):
    model = RandomizedSearchCV(estimator, params, n_jobs=-1, scoring='accuracy',cv=cv, n_iter=n_iter)
    model.fit(train, y)
    return model


# # Modelling

# ## Lasso Regression

# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


logreg = LogisticRegression(penalty='l1', dual=False, solver='liblinear', random_state=0)


# In[47]:


import scipy.stats as st


# In[48]:


params_dist = {  
    "max_iter": st.randint(100,300),
    "C": st.uniform(1.1, 3.0)
}
params_grid = {  
    "max_iter": [200, 210, 220],
    "C": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
}


# In[49]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(logreg, params_dist, X_train, y_train)')


# In[50]:


model.best_score_


# In[51]:


model.best_estimator_


# In[52]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[53]:


from sklearn.metrics import accuracy_score


# In[54]:


accuracy_score(y_cv, cvPredicted)


# ## Decision Tree

# In[55]:


from sklearn.tree import DecisionTreeClassifier


# In[56]:


dt = DecisionTreeClassifier(random_state=0)


# In[57]:


params_grid = {
    "max_depth" : np.arange(1, 25, 1)
}


# In[58]:


get_ipython().run_cell_magic('time', '', 'model = trainModelGrid(dt, params_grid, X_train, y_train)')


# In[59]:


model.best_score_


# In[60]:


model.best_estimator_


# In[61]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[62]:


accuracy_score(y_cv, cvPredicted)


# In[63]:


model.cv_results_['mean_test_score']


# In[64]:


np.argmax(model.cv_results_['mean_test_score'])


# In[65]:


sns.lineplot(x=np.arange(1,25,1), y=model.cv_results_['mean_test_score'])


# In[66]:


model.best_estimator_.fit(X_train, y_train)


# In[67]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[68]:


dot_data = StringIO()

export_graphviz(model.best_estimator_, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('../models/dtree-graph.png')
Image(graph.create_png())


# ## Gradient Boosted Trees

# In[69]:


from xgboost.sklearn import XGBClassifier


# In[70]:


xgb = XGBClassifier(random_state=0)


# In[71]:


one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params_dist = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}


# In[72]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(xgb, params_dist, X_train, y_train, cv=20, n_iter=50)')


# In[73]:


model.best_score_


# In[74]:


model.best_estimator_


# In[75]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[76]:


accuracy_score(y_cv, cvPredicted)


# ## Random Forest Classifier

# In[77]:


from sklearn.ensemble import RandomForestClassifier


# In[78]:


rf = RandomForestClassifier(random_state=0)


# In[79]:


params_dist = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40)
}


# In[80]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(rf, params_dist, X_train, y_train)')


# In[81]:


model.best_score_


# In[82]:


model.best_estimator_


# In[83]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[84]:


accuracy_score(y_cv, cvPredicted)


# ## Support Vector Classifier

# In[85]:


from sklearn.svm import LinearSVC


# In[86]:


svc = LinearSVC(dual=False, random_state=0)


# In[87]:


params = {  
    "C": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
}


# In[88]:


params_dist = {  
    "C": st.uniform(1.0, 2.0)
}


# In[89]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(svc, params_dist, X_train, y_train)')


# In[90]:


model.best_score_


# In[91]:


model.best_estimator_


# In[92]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[93]:


accuracy_score(y_cv, cvPredicted)

