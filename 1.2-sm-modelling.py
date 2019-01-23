
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")


# In[32]:


comment = """RUNNING - Modelling with Full Feature Engineered data.
This script will run the model training for full feature engineered data
that I have generated in previous notebook
NOTE: This script needs the output files of 1.1-sm-feature-engineering 
AND it will fail for the downsampled dataset.
"""
print(comment)
from time import sleep
sleep(5)


# # Obtain Data

# In[2]:


from scipy.sparse import csr_matrix, load_npz


# In[3]:


train = load_npz('1.1-sm-feature-engineered-1.npz')
# train = pd.read_csv('1.1-sm-feature-engineered-1.csv')


# In[4]:


train.shape


# In[7]:


labels = pd.read_csv('1.1-sm-feature-engineered-1-labels.csv')


# In[8]:


labels.info()


# In[9]:


labels.stars.values


# ### Modelling label as binary

# Here I am defining the success measure of a restaurant as being a star rating of >= 3.5 (i.e. label >=7)

# In[10]:


labels = labels >= 7


# In[11]:


labels = labels.astype(np.int)


# In[12]:


labels.stars.value_counts()


# # Setup for Modelling

# ## Splitting data

# In[13]:


X = train
y = labels.stars.values


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_cv, y_train, y_cv = train_test_split( 
              X, y, test_size = 0.4, random_state = 42, stratify = y) 


# In[16]:


X_cv, X_test, y_cv, y_test = train_test_split(X_cv, y_cv, test_size=0.1, random_state=42, stratify=y_cv)


# In[17]:


print(X_train.shape)
print(X_cv.shape)
print(X_test.shape)


# In[18]:


y_train


# ## Training Function

# In[19]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[20]:


def trainModelGrid(estimator, params, train, y, cv=5):
    model = GridSearchCV(estimator, params, n_jobs=-1, scoring='accuracy',cv=cv)
    model.fit(train, y)
    return model


# In[21]:


def trainModelRandom(estimator, params, train, y, cv=5, n_iter=5):
    model = RandomizedSearchCV(estimator, params, n_jobs=-1, scoring='accuracy',cv=cv, n_iter=n_iter)
    model.fit(train, y)
    return model


# # Modelling

# ## Lasso Regression

# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


logreg = LogisticRegression(penalty='l1', dual=False, solver='liblinear', random_state=0)


# In[22]:


import scipy.stats as st


# In[23]:


params_dist = {  
    "max_iter": st.randint(100,300),
    "C": st.uniform(1.1, 3.0)
}
params_grid = {  
    "max_iter": [200, 210, 220],
    "C": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
}


# In[24]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(logreg, params_dist, X_train, y_train)')


# In[25]:


model.best_score_


# In[26]:


model.best_estimator_


# In[27]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


accuracy_score(y_cv, cvPredicted)


# ## Decision Tree

# In[22]:


from sklearn.tree import DecisionTreeClassifier


# In[23]:


dt = DecisionTreeClassifier(random_state=0)


# In[24]:


params_grid = {
    "max_depth" : np.arange(1, 25, 1)
}


# In[25]:


get_ipython().run_cell_magic('time', '', 'model = trainModelGrid(dt, params_grid, X_train, y_train)')


# In[34]:


model.best_score_


# In[35]:


model.best_estimator_


# In[36]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[37]:


accuracy_score(y_cv, cvPredicted)


# In[38]:


model.cv_results_['mean_test_score']


# In[27]:


sns_plot = sns.lineplot(x=np.arange(1,25,1), y=model.cv_results_['mean_test_score'])
sns_plot.figure.savefig('decision-tree-depth-vs-score.png')


# In[28]:


model.best_estimator_.fit(X_train, y_train)


# In[29]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[30]:


dot_data = StringIO()

export_graphviz(model.best_estimator_, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('dtree-graph-full-dataset.png')
Image(graph.create_png())


# ## Gradient Boosted Trees

# In[44]:


from xgboost.sklearn import XGBClassifier


# In[45]:


xgb = XGBClassifier(random_state=0)


# In[46]:


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


# In[47]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(xgb, params_dist, X_train, y_train, cv=20, n_iter=50)')


# In[48]:


model.best_score_


# In[49]:


model.best_estimator_


# In[50]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[51]:


accuracy_score(y_cv, cvPredicted)


# ## Random Forest Classifier

# In[52]:


from sklearn.ensemble import RandomForestClassifier


# In[53]:


rf = RandomForestClassifier(random_state=0)


# In[54]:


params_dist = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40)
}


# In[55]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(rf, params_dist, X_train, y_train)')


# In[56]:


model.best_score_


# In[57]:


model.best_estimator_


# In[58]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[59]:


accuracy_score(y_cv, cvPredicted)


# ## Support Vector Classifier

# In[60]:


from sklearn.svm import LinearSVC


# In[61]:


svc = LinearSVC(dual=False, random_state=0)


# In[62]:


params = {  
    "C": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
}


# In[63]:


params_dist = {  
    "C": st.uniform(1.0, 2.0)
}


# In[64]:


get_ipython().run_cell_magic('time', '', 'model = trainModelRandom(svc, params_dist, X_train, y_train)')


# In[65]:


model.best_score_


# In[66]:


model.best_estimator_


# In[67]:


cvPredicted = model.best_estimator_.predict(X_cv)


# In[68]:


accuracy_score(y_cv, cvPredicted)

