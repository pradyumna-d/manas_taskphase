#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# In[2]:


dataset = pd.read_csv("linear_regression_dataset.csv")
x = dataset.iloc[:, :].values
x = pd.DataFrame(x)
x = x.drop(4, axis=1)
x = np.array(x)
x = np.nan_to_num(x)
y = dataset.iloc[:, -2].values


# In[3]:


x = np.vstack((np.ones((x.shape[0], )), x.T)).T
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
y_train = np.reshape(y_train, (y_train.shape[0],1))
x_test.shape
y_train.shape


# In[4]:


def linear_model(x, y, learning_rate, iteration):
  m = x.shape[0]
  theta = np.zeros((x.shape[1], 1))
  cost_list = []
  for i in range(iteration):
    y_pred = np.dot(x, theta)
    print()
    #cost function
    cost = (1/(2*m))*np.sum((y_pred - y)**2)
    #gradient descent
    d_theta = (1/m) * x.T.dot(y_pred - y)
    theta = theta - learning_rate * d_theta
    cost_list.append(cost)
  return theta, cost_list


# In[5]:


theta, cost_list = linear_model(x_train, y_train, learning_rate=0.00000009, iteration=2000)


# In[6]:


range = np.arange(0, 2000) 

plt.plot(range, cost_list)
plt.show


# In[7]:


predictions = np.dot(x_test, theta)
print(predictions.shape)
print(y_test.shape)
print(x_test.shape)


# In[8]:


predictions


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




