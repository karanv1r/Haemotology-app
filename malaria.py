#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow_hub as hub


# In[ ]:





# In[ ]:





# In[2]:


def testmalaria(img):
    malaria_model = tensorflow.keras.models.load_model("./models/malaria.h5")
    image_shape = (130,130,3)
    new_image = tensorflow.keras.preprocessing.image.load_img(img, target_size=image_shape)
    new_image = image.img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)
    result = malaria_model.predict(new_image)
    if(result[0]):
        return "positive for malaria"
    else:
        return "negative for malaria"
    


# In[ ]:





# In[ ]:





# In[5]:



# In[6]:





# In[7]:





# In[ ]:




