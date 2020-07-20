#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:15:54 2020

@author: NDH00360
"""
import sklearn
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 

import pickle


from sklearn.preprocessing import LabelEncoder




from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

































# In[ ]:





# In[4]:


'''
This is the program for prediciton of category of attack from UNSW dataset

'''
df=pd.read_csv(r"features_having_most_influence_on_UNSW.csv")
df=df[:200000]



## We will drop all the unnessary column 
df1=df.drop(['proto', 'saddr','pkSeqID','daddr','sport','dport'], axis=1)


## We use label encoder to encode categorical data
lb_make = LabelEncoder()
df1['subcategory'] = lb_make.fit_transform(df1['subcategory'])

replace_map = {'category': {'DoS': 1, 'DDoS': 2, 'Normal': 3, 'Theft': 4,'Reconnaissance':5}}

df1.replace(replace_map, inplace=True)


##saving in different dataframe
df2=df1

##obtaining test and train set
X = df2.drop('category', axis=1)
y = df2['category']
X1 = df2.drop('category', axis=1)


## scalling data for faster traning set and accuray on SVM poko
scaler = StandardScaler()

X=scaler.fit_transform(X)
print(X)
print(type(X))




# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[7]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)


# In[8]:


print(type(X))


# In[9]:


relevence_scores=(model.feature_importances_)
print((relevence_scores))


# In[10]:


feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(14).plot(kind='barh')
plt.show()


# In[ ]:


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# In[2]:


##Saving the model into pickle file
filename = 'UNSW_SVM_model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))


# In[9]:


y_pred = svclassifier.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[4]:


##Loadning the pickle file

loaded_model = pickle.load(open('UNSW_SVM_model.sav', 'rb'))

#df_test=pd.read_csv(r"C:\Users\NDH00360\Desktop\TestExe\features_having_most_influence_on_UNSW.csv")

result = loaded_model.score(X_test, y_test)
print("Overall score=",result)

