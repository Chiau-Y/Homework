#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv('./hw4_train.csv') # donwload the train data
X = data.iloc[:,0:data.shape[1]-1] # X_train
y = data.Toughness # y_train

X_test = pd.read_csv('./hw4_test.csv') # donwload the test data


# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0,test_size=0.2) # 20% for validation


# In[3]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # mean to zero, standard deviation to 1
scaler.fit(np.array(y_train).reshape(-1, 1)) # set the model according to every feature respectively

y_train_scaler = scaler.transform(np.array(y_train).reshape(-1, 1)) # StandardScaler
y_val_scaler = scaler.transform(np.array(y_val).reshape(-1, 1)) # StandardScaler


# In[4]:


# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=250, random_state=0)
model.fit(X_train,y_train_scaler) # train

# (1 - u/v), u=((y_true - y_pred) ** 2).sum(), v=((y_true - y_true.mean()) ** 2).sum()
testing_score = model.score(X_val,y_val_scaler) 
print('testing scores : ',testing_score)


# In[5]:


from sklearn.metrics import mean_squared_error

y_val_pred = model.predict(X_val) # predict validation

print('MSE : ',mean_squared_error(y_val_scaler,y_val_pred)) # calculate MSE


# In[6]:


y_pred = model.predict(X_test) # predict

y_pred = scaler.inverse_transform(y_pred) # inverse StandardScaler

y_pred_pd = pd.DataFrame(data=y_pred, columns=['Toughness'])
y_pred_pd = y_pred_pd.reset_index()
y_pred_pd.to_csv('n96084094_HW4_1.csv',index=False) # save the data


# In[7]:


# Fully connected (Dense)
from keras import models
from keras import layers

model = models.Sequential() # set model
model.add(layers.Dense(16, activation='relu', input_shape=(64,))) # 16 output 
model.add(layers.Dense(16, activation='relu')) # 16 output 
model.add(layers.Dense(1, activation='linear')) # 1 output, y = a(wx + b), a = 1


# In[8]:


model.compile(optimizer='rmsprop',
              loss='mean_squared_error', # regression problems
              metrics=['mse']) # regression problems, MSE, MAE, MAPE, Cosine, not accuracy


# In[9]:


history = model.fit(X_train, y_train_scaler, epochs=250, batch_size=1000, validation_data=(X_val, y_val_scaler))


# In[10]:


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1) # x-axis

plt.plot(epochs, loss, 'b', label='Training loss')  # y-axis, loss
plt.plot(epochs, val_loss, 'r', label='Validation loss') # y-axis, , loss
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[11]:


y_pred = model.predict(X_test)

y_pred=scaler.inverse_transform(y_pred)  # inverse StandardScaler

y_pred_pd = pd.DataFrame(data=y_pred, columns=['Toughness'])
y_pred_pd = y_pred_pd.reset_index()
y_pred_pd.to_csv('n96084094_HW4_2.csv',index=False)


# In[ ]:




