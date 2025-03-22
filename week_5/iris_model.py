#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Required Libraries
import pandas as pd
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# Load the Iris dataset
iris = datasets.load_iris()


# In[4]:


# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target


# In[5]:


# Display first few rows
df.head()


# Train the Machine Learning Model

# In[6]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[iris.feature_names], df["target"], test_size=0.2, random_state=42
)


# In[7]:


# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[8]:


# Make predictions
y_pred = model.predict(X_test)


# In[9]:


# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Save the Trained Model

# In[16]:


# Save the trained model as a .pkl file
joblib.dump(model, "iris_model.pkl")

print("Model saved successfully!")


# Load and Test the Model 

# In[11]:


# Load the saved model
loaded_model = joblib.load("iris_model.pkl")


# In[ ]:


# Test with sample data
sample_input = [[4.9,3.0,1.4,0.2]] 
prediction = loaded_model.predict(sample_input)


# In[15]:


print(f"Predicted Class: {prediction[0]} ({iris.target_names[prediction[0]]})")


# In[ ]:




