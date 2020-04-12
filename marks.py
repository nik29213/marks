#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


dfx = pd.read_csv("Linear_X_Train.csv")
dfy = pd.read_csv("Linear_Y_Train.csv")
x = dfx.values
y = dfy.values
plt.scatter(x,y)


# In[3]:


X,Y = (x-x.mean())/x.std(),y
plt.scatter(X,Y)


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


lr = LinearRegression(normalize=True)
lr.fit(X,Y)


# In[6]:


lr.predict([[10]])


# In[9]:


plt.scatter(X,Y)
plt.plot(X,lr.predict(X),color="orange")


# In[10]:


from sklearn.externals import joblib


# In[11]:


joblib.dump(lr,"model.pkl")
m=joblib.load("model.pkl")


# In[ ]:




