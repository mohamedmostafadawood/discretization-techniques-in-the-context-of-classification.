#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score


# In[2]:


test = pd.read_csv("Test.csv")
train = pd.read_csv("Train.csv")


# In[3]:


def equal_interval_disc(data_set , num_of_intervals) :
    cutted_data_set=pd.cut(data_set , num_of_intervals,labels=False)
    return cutted_data_set    


# In[4]:


def equal_freq_disc (data_set , num_of_freq):
    new_data_set = pd.qcut(data_set , num_of_freq,labels=False)
    return new_data_set


# In[7]:


def eucliden_distance (point1 , point2 ):
    return np.linalg.norm(point1 - point2)


# In[8]:


merged_dataset = pd.concat([train,test],ignore_index=True)


# In[52]:


def predictions (disc_type , n):
    if disc_type=="equal_interval_disc" :
        disc_length=equal_interval_disc(merged_dataset["PetalLengthCm"],n)
        disc_width=equal_interval_disc(merged_dataset["PetalWidthCm"],n)
    else :
        
        disc_length=equal_freq_disc(merged_dataset["PetalLengthCm"],n)
        disc_width=equal_freq_disc(merged_dataset["PetalWidthCm"],n)
        
        
    merged_dataset["LengthDisc"] = disc_length
    merged_dataset["WidthDisc"] = disc_width    
    
    disc_train=merged_dataset[:train["Species"].size]
    
    setosa_train=disc_train.loc[disc_train["Species"]== "Iris-setosa"]
    virginica_train = disc_train.loc[disc_train["Species"] == "Iris-virginica"]
    versicolor_train = disc_train.loc[disc_train["Species"] == "Iris-versicolor"]
    
    
    versicolor_mean = (versicolor_train["LengthDisc"].mean(),versicolor_train["WidthDisc"].mean())
    virginica_mean = (virginica_train["LengthDisc"].mean(),virginica_train["WidthDisc"].mean())                            
    setosa_mean = (setosa_train["LengthDisc"].mean(),setosa_train["WidthDisc"].mean())
    
    disc_test = merged_dataset[train["Species"].size : ].copy()
    
    
    test_length_width=[]
    for ind in disc_test.index:
        test_length_width.append((disc_test['LengthDisc'][ind], disc_test['WidthDisc'][ind]))
     
    
    species_predicts=[]
    
    for item in (test_length_width) :
        setosa_distance = eucliden_distance(np.array(item) , np.array(setosa_mean))
        versicolor_distance=eucliden_distance(np.array(item) , np.array(versicolor_mean))
        virginica_distance=eucliden_distance(np.array(item) , np.array(virginica_mean))


        if setosa_distance<versicolor_distance and setosa_distance<virginica_distance:
            species_predicts.append("Iris-setosa")
        elif versicolor_distance<setosa_distance and versicolor_distance<virginica_distance:
            species_predicts.append("Iris-versicolor")

        elif virginica_distance<setosa_distance and virginica_distance<versicolor_distance:
            species_predicts.append("Iris-virginica") 
            
            
    disc_test["Species"] = species_predicts
    return disc_test


# In[53]:


def accuracy (train_result , test_result):
    score = accuracy_score(train_result,test_result)


# In[54]:


def classification_accuracy ( actual , predicted ):
    correct = 0
    actual_items=list(actual["Species"])
    predicted_items=list(predicted["Species"])
    for i in range(len(actual_items)):
        if actual_items[i] == predicted_items[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
    


# In[63]:


#predicted_set=predictions("equal_interval_disc",2)


equal_interval_result1=classification_accuracy(test,predictions("equal_interval_disc",2))
equal_interval_result2=classification_accuracy(test,predictions("equal_interval_disc",3))

equal_interval_freq1=classification_accuracy(test,predictions("",2))
equal_interval_freq2=classification_accuracy(test,predictions("",3))

print ("The classification accuracy achieved when using Equal Interval discretization with n = 2 intervals: "+ " "+ str(equal_interval_result1))
print ("The classification accuracy achieved when using Equal Interval discretization with n = 3 intervals: "+ " "+ str(equal_interval_result2))
print ("The classification accuracy achieved when using Equal Frequency discretization with n = 2 intervals "+ " "+ str(equal_interval_freq1))
print ("The classification accuracy achieved when using Equal Frequency discretization with n = 3 intervals "+ " "+ str(equal_interval_freq2))


# In[ ]:




