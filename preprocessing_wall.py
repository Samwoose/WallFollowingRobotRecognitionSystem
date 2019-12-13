#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import random
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#########################Collection of functions##############################



def balancedDataSetCreater(df_numeric_label):
    '''
    input:
    1. df_numeric_label : (unbalanced) data set with numeric label , type: DataFrame
    
    output:
    1. bal_df : balanced data set, type: DataFrame

    '''
    df_class1 = pd.DataFrame({'f1' : [] ,'f2' : [], 'f3' : [] ,'f4' : [],'f5' : [] ,'f6' : [],'f7' : [] ,'f8' : [],'f9' : [] ,'f10' : [],'f11' : [] ,'f12' : [],'f13' : [] ,'f14' : [],'f15' : [] ,'f16' : [],'f17' : [] ,'f18' : [],'f19' : [] ,'f20' : [], 'f21' : [] ,'f22' : [],'f23' : [] ,'f24' : [] ,'label' : []})
    df_class2 = pd.DataFrame({'f1' : [] ,'f2' : [], 'f3' : [] ,'f4' : [],'f5' : [] ,'f6' : [],'f7' : [] ,'f8' : [],'f9' : [] ,'f10' : [],'f11' : [] ,'f12' : [],'f13' : [] ,'f14' : [],'f15' : [] ,'f16' : [],'f17' : [] ,'f18' : [],'f19' : [] ,'f20' : [], 'f21' : [] ,'f22' : [],'f23' : [] ,'f24' : [] ,'label' : []})
    df_class3 = pd.DataFrame({'f1' : [] ,'f2' : [], 'f3' : [] ,'f4' : [],'f5' : [] ,'f6' : [],'f7' : [] ,'f8' : [],'f9' : [] ,'f10' : [],'f11' : [] ,'f12' : [],'f13' : [] ,'f14' : [],'f15' : [] ,'f16' : [],'f17' : [] ,'f18' : [],'f19' : [] ,'f20' : [], 'f21' : [] ,'f22' : [],'f23' : [] ,'f24' : [] ,'label' : []})
    df_class4 = pd.DataFrame({'f1' : [] ,'f2' : [], 'f3' : [] ,'f4' : [],'f5' : [] ,'f6' : [],'f7' : [] ,'f8' : [],'f9' : [] ,'f10' : [],'f11' : [] ,'f12' : [],'f13' : [] ,'f14' : [],'f15' : [] ,'f16' : [],'f17' : [] ,'f18' : [],'f19' : [] ,'f20' : [], 'f21' : [] ,'f22' : [],'f23' : [] ,'f24' : [] ,'label' : []})

    for index in range(0,len(df_numeric_label)):
        if(df_numeric_label.loc[index,'label'] == 0):
            #class1
            df_class1 = df_class1.append(df_numeric_label.loc[index])
        elif(df_numeric_label.loc[index,'label'] == 1):
            #class2
            df_class2 = df_class2.append(df_numeric_label.loc[index]) 
        elif(df_numeric_label.loc[index,'label'] == 2):
            #class3
            df_class3 = df_class3.append(df_numeric_label.loc[index])
        elif(df_numeric_label.loc[index,'label'] == 3):
            #class4
            df_class4 = df_class4.append(df_numeric_label.loc[index])     
        
    #convert index in order
    df_class1 = pd.DataFrame(np.array(df_class1))
    df_class2 = pd.DataFrame(np.array(df_class2))
    df_class3 = pd.DataFrame(np.array(df_class3))
    df_class4 = pd.DataFrame(np.array(df_class4))

    ##Seperate X and y
    #24 means 'label'
    #Seperate feature data points 
    X_class1 = df_class1.drop( 24 ,axis = 1) 
    X_class2 = df_class2.drop( 24 ,axis = 1) 
    X_class3 = df_class3.drop( 24 ,axis = 1) 
    #Seperate class labels
    y_class1=pd.DataFrame(df_class1.loc[:,24])
    y_class2=pd.DataFrame(df_class2.loc[:,24])
    y_class3=pd.DataFrame(df_class3.loc[:,24])


    #Down sample data points of class1,2,3 to the same number of data points of class4
    X_class1_down , temp_X_test, y_class1_down , temp_y_test = train_test_split(X_class1, y_class1, train_size = len(df_class4))
    X_class2_down , temp_X_test, y_class2_down , temp_y_test = train_test_split(X_class2, y_class2, train_size = len(df_class4))
    X_class3_down , temp_X_test, y_class3_down , temp_y_test = train_test_split(X_class3, y_class3, train_size = len(df_class4))

    #thanks to https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    #How to combine two DataFrames
    #combine feature data points and class label
    df_class1_down = pd.concat([X_class1_down, y_class1_down] , axis = 1, sort = False)
    df_class2_down = pd.concat([X_class2_down, y_class2_down] , axis = 1, sort = False)
    df_class3_down = pd.concat([X_class3_down, y_class3_down] , axis = 1, sort = False)

    #comebine each data point from class1 , 2, 3, 4
    df_data_set_bal = pd.concat([df_class1_down, df_class2_down, df_class3_down, df_class4 ])
    
    return df_data_set_bal
 



################### Main part #########################


df = pd.read_csv('sensor_readings_24-Copy1.data')

#assign pre space for data frame that has numeric labels
df_numeric_label = df

#Convert categorical label names to numeric labels
for index in range(0,len(df_numeric_label)):
    if(df_numeric_label.loc[index,'label'] == 'Slight-Right-Turn'):
        df_numeric_label.loc[index,'label'] = 0 #class1
    elif(df_numeric_label.loc[index,'label'] == 'Move-Forward'):
        df_numeric_label.loc[index,'label'] = 1 #class2
    elif(df_numeric_label.loc[index,'label'] == 'Sharp-Right-Turn'):
        df_numeric_label.loc[index,'label'] = 2 #class3
    elif(df_numeric_label.loc[index,'label'] == 'Slight-Left-Turn'):
        df_numeric_label.loc[index,'label'] = 3 #class4

#Seperate feature data points
X = df_numeric_label.drop('label',axis = 1)        
#Seperate class labels
y=pd.DataFrame(df_numeric_label.loc[:,'label'])

#split training data set and test data set 
#80% of D for D'', 20% of D for D_test with rule of thumb
X_tr_80 , X_te_20, y_tr_80, y_te_20 = train_test_split(X, y, train_size=8/10)
X_tr_80_arr = np.array(X_tr_80)
y_tr_80_arr = np.array(y_tr_80)


#Split D_test to D_val_te and D_fin_te
#50% of D_test for each data set
X_val_te_50 , X_fin_te_50 , y_val_te_50 , y_fin_te_50 = train_test_split(X_te_20, y_te_20 , train_size = 1/2)
X_val_te_50_arr = np.array(X_val_te_50) 
X_fin_te_50_arr = np.array(X_fin_te_50) 
y_val_te_50_arr = np.array(y_val_te_50) 
y_fin_te_50_arr = np.array(y_fin_te_50)


##Create Balanced Data
D_dot_dot = pd.DataFrame(np.array(pd.concat([X_tr_80, y_tr_80] , axis = 1, sort = False)))
#thanks to code from https://cmdlinetips.com/2018/03/how-to-change-column-names-and-row-indexes-in-pandas/
#How to change name of columns
D_dot_dot.columns = ['f1' ,'f2' , 'f3' ,'f4' ,'f5' ,'f6' ,'f7' ,'f8' ,'f9' ,'f10' ,'f11' ,'f12' ,'f13' ,'f14' ,'f15' ,'f16' ,'f17' ,'f18' ,'f19' ,'f20' , 'f21' ,'f22' ,'f23'  ,'f24' ,'label']
df_data_set_bal = balancedDataSetCreater(D_dot_dot)
#df_data_set_bal = balancedDataSetCreater(D_dot_dot)

#Seperate X and y
X_bal = df_data_set_bal.drop( 24 ,axis = 1) 
y_bal = pd.DataFrame(df_data_set_bal.loc[:,24])

X_tr_bal_arr = np.array(X_bal)
y_tr_bal_arr = np.array(y_bal)



#save data files
#1. whole data set
df_numeric_label.to_csv(r'data_wall\data_wall.csv')

#2. D''
D_dot_dot.to_csv(r'data_wall\data_train.csv')

#.3. D''_bal
df_data_set_bal.to_csv(r'data_wall\data_train_bal.csv')

#4. D_val_test
#combine X, y first
D_val_test = pd.DataFrame(np.array(pd.concat([X_val_te_50, y_val_te_50] , axis = 1, sort = False)))
D_val_test.to_csv(r'data_wall\data_val_test.csv')               

#5. D_fin_test
D_fin_test = pd.DataFrame(np.array(pd.concat([X_fin_te_50, y_fin_te_50] , axis = 1, sort = False)))
D_fin_test.to_csv(r'data_wall\data_fin_test.csv') 
 

