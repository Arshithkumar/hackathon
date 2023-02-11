#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1
data = pd.read_csv("dataframe_ - dataframe_.csv")


# In[41]:


# Step 2:
print(data.isnull().sum())

sns.pairplot(data)
plt.show()

for col in data.columns:
    sns.boxplot(data[col])
    plt.show()


# In[42]:


# Step 3:
X = data.drop(columns=['output'])
y = data['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[43]:


# Step 4: Outlier Detection and Treatment
from scipy import stats

z_scores = np.abs(stats.zscore(X_train))
X_train_scaled = X_train_scaled[(z_scores < 3).all(axis=1)]
y_train = y_train[(z_scores < 3).all(axis=1)]

# Step 4: Hyperparameter Tuning
# Random Forest Regressor
rf = RandomForestRegressor()
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [2, 4, 6]}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)


# In[24]:



# Step 5: Evaluation Metrics and Model Comparison
# Evaluate the Random Forest Regressor on the test data
y_pred = grid_search.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Random Forest Regressor:")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# In[25]:


# Step 6: Bonus - Build a Linear Regression Model
# Feature Engineering
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly


# In[ ]:




