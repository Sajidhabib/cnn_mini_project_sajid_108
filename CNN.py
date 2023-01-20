#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow as tf
import pandas as pd 
import numpy as np


# In[12]:


# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\dell\Desktop\training_set\training_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=64,
    image_size=(224, 224)
)


# In[13]:


dataset


# In[14]:


# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)


# In[15]:


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(2, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(5, activation='softmax')  # 5 classes
        self.dropout = tf.keras.layers.Dropout(0.25)

    def call(self, inputs, training=False):
        x = self.pool(self.conv1(inputs))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


# In[16]:


# Create the model
model = CNN()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[17]:


# Train the model
model = model.fit(train_dataset,epochs=5)


# In[ ]:




