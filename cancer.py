#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[10]:


# data analysis
import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# utilities
import os
from tqdm import tqdm

# machine learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub


# In[81]:





# In[ ]:





# In[ ]:





# In[82]:





# In[11]:


def process_image(image):
    IMG_SIZE = 128
    img = tf.io.read_file(image)
    img = tf.io.decode_bmp(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE,IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    plt.imshow(img)
    return img


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


def testcancer(path):
    IMG_SIZE = 128
    image = path
    input_scalar = tf.reshape(image, [])
    img = process_image(input_scalar)
    image = np.expand_dims(img, axis=0)
    print(image.shape)
    new_model = tf.keras.models.load_model(('./models/cancer_detector.h5'),custom_objects={'KerasLayer':hub.KerasLayer})
    prediction = new_model.predict(image)
    prediction = np.argmax(prediction,axis=1)
    if (prediction == 0):
        return "positive for acute lymphocytic luekemia"
    elif (prediction == 1):
        return "negative for acute lymphocytic luekemia"


# In[ ]:





# In[ ]:




