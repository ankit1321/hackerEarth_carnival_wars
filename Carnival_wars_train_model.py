#!/usr/bin/env python
# coding: utf-8

# In[312]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,explained_variance_score,r2_score
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import layers
from scipy import stats
import math
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from mpl_toolkits import mplot3d
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


# In[5]:


data = pd.read_csv("./dataset/train.csv")


# In[6]:


data.head()


# In[7]:


data.info()


# In[429]:


price_prediction_feature= data.copy()


# In[430]:


price_prediction_feature.isna().sum()


# In[431]:


price_prediction_feature.shape


# In[432]:


labels = price_prediction_feature.pop("Selling_Price")


# In[433]:


price_prediction_feature = price_prediction_feature.fillna(price_prediction_feature.mean())


# In[434]:


price_prediction_feature["Selling_Price"] = labels
type(price_prediction_feature),price_prediction_feature.shape


# In[435]:


labels = pd.concat([price_prediction_feature["Minimum_price"],price_prediction_feature["Maximum_price"],
                    price_prediction_feature["Selling_Price"]],axis=1)
labels.shape,labels.head()


# In[436]:


labels.isna().sum(),type(labels)


# In[437]:


modified = []
for i,j,k in zip(labels["Maximum_price"],labels["Minimum_price"],labels["Selling_Price"]):
    if k > 0:
        modified.append([i,j,k])
    else:
        modified.append([i,j,(i+j)/2])


# In[438]:


modified = np.array(modified)
_ = price_prediction_feature.pop("Selling_Price")
_ = price_prediction_feature.pop("Maximum_price")
_ = price_prediction_feature.pop("Minimum_price")
modified.shape,price_prediction_feature.shape


# In[439]:


labels = pd.concat([price_prediction_feature,pd.DataFrame(modified)],axis=1)
type(labels)


# In[440]:


labels.isna().sum()


# In[441]:


labels


# In[442]:


price_prediction_feature = labels.copy()


# In[443]:


price_prediction_feature = price_prediction_feature.dropna()


# In[444]:


price_prediction_feature.shape


# In[445]:


price_prediction_feature.isna().sum()


# In[446]:


out = price_prediction_feature.pop(2)


# In[447]:


price_prediction_feature.info()


# In[448]:


_ = price_prediction_feature.pop("Product_id")


# In[449]:


price_prediction_feature.info()


# In[450]:


price_prediction_feature = np.array(price_prediction_feature)
labels = np.array(labels)


# In[451]:


encoder = OrdinalEncoder()


# In[452]:


price_prediction_feature = encoder.fit_transform(price_prediction_feature)
price_prediction_feature


# In[453]:


price_prediction_feature = preprocessing.normalize(price_prediction_feature)
price_prediction_feature


# In[454]:


price_prediction_feature.shape


# In[455]:


price_prediction_feature = preprocessing.minmax_scale(price_prediction_feature)
price_prediction_feature


# In[456]:


maximum,minimum = max(out),min(out)
maximum,minimum


# In[457]:


out = (out-minimum)/(maximum-minimum)


# In[ ]:





# In[458]:


X_train,X_val,Y_train,Y_val = train_test_split(price_prediction_feature,out,test_size=0.1,random_state=0)
print(price_prediction_feature.shape,out.shape,X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)


# In[459]:


used_cars_model = tf.keras.Sequential([
    layers.Dense(2048,activation="relu"),
    layers.Dense(1096,activation="relu"),
    layers.Dense(8,activation="tanh"),
    layers.Dense(1)
])
monitor = ModelCheckpoint("./checkpoint/",verbose=1,save_best_only=True)
used_cars_model.compile(loss = tf.losses.MeanSquaredError(),optimizer = tf.optimizers.Adam(learning_rate=0.001))


# In[489]:


Epoch=200
BS = 36
H = used_cars_model.fit(X_train,Y_train,batch_size=BS,steps_per_epoch=len(X_train)//BS,callbacks=[monitor],verbose=2,
                        validation_data=(X_val, Y_val),validation_steps=len(X_val)//BS,epochs = Epoch)


# In[490]:


from tensorflow.keras.models import load_model
model = load_model("./checkpoint/")


# In[491]:


model.evaluate(X_val,Y_val)


# In[492]:


#Get predicted value from validation data 
Y_predicted =  model.predict(X_val)
print("predicted array",Y_predicted)


# In[493]:


#Match the dimensions of true and predicted value
print(Y_predicted.shape,Y_val.shape)
Y_predicted = np.reshape(Y_predicted,(Y_val.shape))
print(Y_predicted.shape,Y_val.shape)


# In[494]:


print("Prediction Summary scaled")
print("Mean Squared Error",mean_squared_error(Y_val,Y_predicted))
print("RMSE",sqrt(mean_squared_error(Y_val,Y_predicted)))
print("R2 score =", round(r2_score(Y_val, Y_predicted), 2))
print("R2 score near to 1 means model is well fitted")
print("Explain variance score =", round(explained_variance_score(Y_val, Y_predicted), 4))


# In[484]:


original_Y_val = Y_val*(maximum-minimum)+minimum
original_Y_pred = Y_predicted*(maximum-minimum)+minimum


# In[485]:


print("prediction summary original")
print("Mean Squared Error",mean_squared_error(original_Y_val,original_Y_pred))
print("RMSE",sqrt(mean_squared_error(original_Y_val,original_Y_pred)))
print("R2 score =", round(r2_score(original_Y_val,original_Y_pred), 2))
print("R2 score near to 1 means model is well fitted")
print("Explain variance score =", round(explained_variance_score(original_Y_val,original_Y_pred), 4))


# In[487]:


# checking for some real and predicted values
for i,j in zip(original_Y_val[:10],original_Y_pred[:10]):
    print("True Value",round(i,2),"       Predicted Value",round(j,2))


# In[488]:


model.save("Best_model",save_format="h5")


# In[ ]:




