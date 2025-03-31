#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


# In[2]:


# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target


# In[3]:


# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Train the model
model = LogisticRegression(max_iter=2000, solver='saga')
model.fit(X_train, y_train)


# In[6]:


# Save the trained model
joblib.dump(model, "iris_model.pkl")


# In[7]:


print("Model trained and saved successfully!")


# In[ ]:




