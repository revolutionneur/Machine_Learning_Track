#!/usr/bin/env python
# coding: utf-8

# In[25]:


import streamlit as st
import numpy as np
import joblib


# In[26]:


model=joblib.load('iris_svc.pkl')
scaler=joblib.load('scaler.pkl')


# In[ ]:





# In[27]:


st.title('My first streamlit app')
st.header('inter the flowers parameters')

sepal_length = st.slider('Sepal Length', 0.0, 10.0, step=0.1)
sepal_width = st.slider('Sepal Width', 0.0, 10.0, step=0.1)
petal_length = st.slider('Petal Length', 0.0, 10.0, step=0.1)
petal_width = st.slider('Petal Width', 0.0, 10.0, step=0.1)

# Perform prediction using the loaded model
#scaled_features=scaler.transform(np.array[sepal_length, sepal_width, petal_length, petal_width])

prediction = model.predict(scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]]))
labels = ['Setosa', 'Versicolour', 'Virginica']
prediction_str = labels[int(prediction)]

# Display the prediction
st.subheader('Prediction:')
st.write(prediction_str)


# In[ ]:




