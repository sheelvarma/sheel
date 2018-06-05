
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("winequality-red.csv")


# In[3]:


df


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()


# In[8]:


scaler.fit(df.drop('quality',axis=1))


# In[9]:


scaled_features = scaler.transform(df.drop('quality',axis=1))


# In[10]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[11]:


from sklearn.cross_validation import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['quality'], test_size=0.33)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier


# In[14]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[15]:


knn.fit(X_train,y_train)


# In[16]:


pred = knn.predict(X_test)


# In[17]:


from sklearn.metrics import classification_report,confusion_matrix


# In[18]:


print(confusion_matrix(y_test,pred))


# In[19]:


print(classification_report(y_test,pred))


# In[20]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[21]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[29]:


knn = KNeighborsClassifier(n_neighbors=22)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=22')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

