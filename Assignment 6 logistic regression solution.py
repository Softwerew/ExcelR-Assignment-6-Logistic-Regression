#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[2]:


data=pd.read_csv(r"C:\Users\Lovely_Ray\Desktop\data science\Assignment 6\bank-full.csv")


# In[3]:


data


# In[4]:


data.info() #EDA


# In[5]:


data1=pd.get_dummies(data,columns=['job','marital','education','contact','poutcome','month']) #OHO of categorical variables
data1


# In[6]:


pd.set_option("display.max.columns", None) #to view all columns
data1


# In[7]:


data1.info()


# In[8]:


# custom binary encoding of binary o/p variables
data1['default'] = np.where(data1['default'].str.contains("yes"), 1, 0)
data1['housing'] = np.where(data1['housing'].str.contains("yes"), 1, 0)
data1['loan'] = np.where(data1['loan'].str.contains("yes"), 1, 0)
data1['y'] = np.where(data1['y'].str.contains("yes"), 1, 0)
data1


# In[9]:


data1.info()


# In[10]:


# dividing data into input and output variable
x=pd.concat([data1.iloc[:,0:10],data1.iloc[:,11:]],axis=1)
y=data1.iloc[:,10]


# In[11]:


#Logistic regression and fit the model
classifier = LogisticRegression()
classifier.fit(x,y)


# In[12]:


#Predict for x dataset
y_pred = classifier.predict(x)
y_pred


# In[13]:


y_pred_df= pd.DataFrame({'actual': y,
                         'predicted_prob': classifier.predict(x)})


# In[14]:


y_pred_df


# In[15]:


# Confusion Matrix for the model accuracy
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix)


# In[16]:


# The model accuracy is calculated by (a+d)/(a+b+c+d)
(39156+1163)/(39156+766+4126+1163)*100


# In[17]:


print(classification_report(y,y_pred)) #Classification report


# In[19]:


#ROC Curve plotting and calculating AUC value
fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba (x)[:,1])

auc = roc_auc_score(y, y_pred)

plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# In[20]:


auc


# In[ ]:


#AUC accuracy rate is 60%.

