#!/usr/bin/env python
# coding: utf-8

# ## Implement the Linear regression on the salary Prediction based on a given set of training data samples.Read the training data from .cse file

# By calling warnings.filter, you are telling Python to ignore any warning messages that might be generated during the execution
# of the code. This can be useful if you know that the warning

# In[45]:


import warnings
warnings.filterwarnings('ignore')


# ## sklearn.model_selection:A module is scikit-learn library that provides various functions for spliting the dataset into train
# ## statsmodels.api:

# In[73]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[47]:


#import data
data = pd.read_csv('Salary_Data.csv')
data.head(10)


# Let us inspect the data:

# In[48]:


data.shape


# we hardly have 30 data points.

# In[49]:


data.describe()


# Min years of experience is 1.1 with as salary of 27414.4 and max years of experience is 10.5 with a salary of 122391.

# Let us visualise the data:

# In[50]:


sns.pairplot(y_vars = 'Salary', x_vars = 'YearsExperience' ,data = data)


# Salary looks linear related to years  of experience

# data.corr() calculates the pairwise correlation between all the columns in a pandas DataFrame data. This correlation matrix 
# can be useful for identifying the strength and direction of the relationships between variables in the dataset. it can also be 
# vesualized using heatmaps or other graphical tools to make ti easier to interpret

# In[51]:


#checking the correlation od the data
data.corr()


# 0.97 is highly correlated

#  Data predection:

# In[52]:


X = data['YearsExperience']
y = data['Salary']


# In[53]:


X_train,X_test,y_train,y_test = train_test_split(X,y, train_size = 0.7,test_size=0.3, random_state = 100)


# In[54]:


X_train.shape


# In[55]:


X_test.shape


# Model Building:

# Implemet ordinary least squares (OLS) linear regression using the statsmodels library:
#  
# x_train_sm=sm.add_constant(X_train):This adds a column if 1st to the x_Train DataFrame ,which serves as the intercept team in the regression model. 

# In[56]:


X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()


# In[57]:


print(model.summary())


# So our line regression equation is:
#     

# Salary=25200+TYearsExperiencex9731.2038

# In[58]:


#let us show the line fitting:
plt.scatter(X_train,y_train)
plt.plot(X_train,25200 + X_train * 9731.2038,'r')
plt.show()


# Residual analysis:
# 

# In[59]:


y_train_pred = model.predict(X_train_sm)


# In[60]:


y_train_pred.head()


# In[61]:


residual = (y_train -y_train_pred)


# In[62]:


residual.head()


# In[63]:


sns.distplot(residual)


# Residual looks normally distributed

# In[64]:


#sns.scatterplot(X_train,residual)
sns.scatterplot(x=X_train ,y=residual)


# Predictions on the test data:

# In[68]:


X_test_sm = sm.add_constant(X_test)


# In[69]:


y_pred = model.predict(X_test_sm)


# Root mean squared error:

# In[74]:


RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
RMSE


# In[ ]:


RMSE wont suggest analy


# R2 score:
#     

# In[75]:


r2_score(y_test,y_pred)


# In[79]:


# let us show the lines fitting:
plt.scatter(X_test,y_test)
plt.plot(X_test, 25200 + X_test * 9731.2038,'r')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




