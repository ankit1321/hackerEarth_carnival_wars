#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import load_model


# In[2]:


test = pd.read_csv("./dataset/test.csv")
test.head()


# In[3]:


data = test.copy()
data.isna().sum()


# In[4]:


range_val = data["Maximum_price"]-data["Minimum_price"]
range_val_mean = range_val.mean()
range_val_mean


# In[5]:


minimum_mean_price = []
for i,j in zip(data["Minimum_price"],data["Maximum_price"]):
    if i > 0:
        minimum_mean_price.append(i)
    else:
        minimum_mean_price.append(j-range_val_mean)


# In[6]:


minimum_mean_price = pd.DataFrame(minimum_mean_price)
minimum_mean_price[1] = data.pop("Maximum_price")
minimum_mean_price.head()


# In[7]:


_ = data.pop("Minimum_price")


# In[8]:


data = data.fillna(data.mean())
data.isna().sum()


# In[9]:


# data = data.replace(np.nan," ")
_ = data.pop("Customer_name")
data.isna().sum()


# In[10]:


product_id = data.pop("Product_id")


# In[11]:


data["mean"] = (minimum_mean_price[0]+minimum_mean_price[1])/2
data.head()


# In[12]:


product_cat = data.pop("Product_Category")


# In[13]:


Grade = data.pop("Grade")


# In[14]:


encoder = preprocessing.OrdinalEncoder()


# In[15]:


product_cat = encoder.fit_transform(np.expand_dims(product_cat,1))
product_cat


# In[16]:


data = encoder.fit_transform(data)


# In[17]:


Grade = encoder.fit_transform(np.expand_dims(Grade,1))


# In[18]:


data.shape


# In[19]:


onehot = preprocessing.OneHotEncoder(sparse=False)


# In[20]:


product_cat = onehot.fit_transform(product_cat)
product_cat.shape


# In[21]:


Grade = onehot.fit_transform(Grade)
Grade.shape


# In[22]:


# data = preprocessing.normalize(data)
# data


# In[23]:


data = preprocessing.minmax_scale(data)
data


# In[24]:


final = pd.concat([pd.DataFrame(data),pd.DataFrame(product_cat),pd.DataFrame(Grade)],axis=1)
final.head()


# In[25]:


final = np.array(final)
final.shape


# In[34]:


model = load_model("Best_model")


# In[35]:


result = model.predict(final)


# In[36]:


result


# In[37]:


count = 0
for i in range(len(result)):
    if minimum_mean_price[0][i]<=result[i]<=minimum_mean_price[1][i]:
        count+=1
count


# In[93]:


submission = pd.DataFrame(product_id)


# In[94]:


submission["Selling_Price"] = result
submission.head()


# In[95]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




