#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


################### Main part #########################
#load D'', D''_bal , D_val_test and D_fin_test
#D''
D_dot_dot = pd.read_csv(r'data_wall\data_train.csv')
X_tr_80 = D_dot_dot.drop('Unnamed: 0' , axis = 1)
X_tr_80 = X_tr_80.drop('label', axis = 1)
y_tr_80 = pd.DataFrame(D_dot_dot.loc[:,'label'])
X_tr_80_arr = np.array(X_tr_80)
y_tr_80_arr = np.array(y_tr_80)

#D''_bal
D_dot_dot_bal = pd.read_csv(r'data_wall\data_train_bal.csv')
X_tr_bal = D_dot_dot_bal.drop('Unnamed: 0' , axis = 1)
X_tr_bal = X_tr_bal.drop('24', axis = 1) #24 means label
y_tr_bal = pd.DataFrame( D_dot_dot_bal.loc[:,'24'])
X_tr_bal_arr = np.array(X_tr_bal)
y_tr_bal_arr = np.array(y_tr_bal)

#D_val_test
D_val_test = pd.read_csv(r'data_wall\data_val_test.csv')
X_val_te_50 = D_val_test.drop('Unnamed: 0' , axis = 1)
X_val_te_50 = X_val_te_50.drop('24', axis = 1) #24 means label
y_val_te_50 = pd.DataFrame( D_val_test.loc[:,'24'])
X_val_te_50_arr = np.array(X_val_te_50)
y_val_te_50_arr = np.array(y_val_te_50)

#D_fin_test
D_fin_test = pd.read_csv(r'data_wall\data_fin_test.csv')
X_fin_te_50 = D_fin_test.drop('Unnamed: 0' , axis = 1)
X_fin_te_50 = X_fin_te_50.drop('24', axis = 1) #24 means label
y_fin_te_50 = pd.DataFrame( D_fin_test.loc[:,'24'])
X_fin_te_50_arr = np.array(X_fin_te_50)
y_fin_te_50_arr = np.array(y_fin_te_50)

##load trained models
#Random Forest models
loaded_model_RF_Unbal_wall = pickle.load(open('fin_RF_clf_unbal_model_wall.sav','rb'))
loaded_model_RF_bal_wall = pickle.load(open('fin_RF_clf_bal_model_wall.sav','rb'))

#L2 models
loaded_model_L2_Unbal_wall = pickle.load(open('fin_LR_cla_L2_unbal_model_wall.sav','rb'))
loaded_model_L2_bal_wall = pickle.load(open('fin_LR_cla_L2_bal_model_wall.sav','rb'))

#L1 models
loaded_model_L1_Unbal_wall = pickle.load(open('fin_LR_cla_L1_unbal_model_wall.sav','rb'))
loaded_model_L1_bal_wall = pickle.load(open('fin_LR_cla_L1_bal_model_wall.sav','rb'))


##calculate E_train, and E_val_test for each model


##Chosen Random Forest Models
print('********************** Random Forest Models************************ \n')
#E_train
predicted_label_on_train_RF = loaded_model_RF_Unbal_wall.predict(X_tr_80_arr)
accuracy_on_train_RF = accuracy_score(y_tr_80_arr, predicted_label_on_train_RF)
print('Accuracy on train data set with RF is : ',accuracy_on_train_RF )


#E_val_test(model1)
predicted_label_on_val_test_RF = loaded_model_RF_Unbal_wall.predict(X_val_te_50_arr)
accuracy_on_val_test_RF = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_RF)
print('Accuracy on validation test data set with RF is : ',accuracy_on_val_test_RF )

print('Selected parameters : \n' , loaded_model_RF_Unbal_wall.get_params()  , '\n')

#E_train_bal
predicted_label_on_train_bal_RF = loaded_model_RF_bal_wall.predict(X_tr_bal_arr)
accuracy_on_train_bal_RF = accuracy_score(y_tr_bal_arr, predicted_label_on_train_bal_RF)
print('Accuracy on balanced train data set with RF is : ',accuracy_on_train_bal_RF )

#E_val_test(model2)
predicted_label_on_val_test_bal_RF = loaded_model_RF_bal_wall.predict(X_val_te_50_arr)
accuracy_on_val_test_bal_RF = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_bal_RF)
print('Accuracy on validation test data set with RF trained by balanced D'' is : ',accuracy_on_val_test_bal_RF )

print('Selected parameters : \n' , loaded_model_RF_bal_wall.get_params()  )



##Chosen L2 models
print('********************** L2 Models************************ \n')
#E_train
predicted_label_on_train_L2 = loaded_model_L2_Unbal_wall.predict(X_tr_80_arr)
accuracy_on_train_L2 = accuracy_score(y_tr_80_arr, predicted_label_on_train_L2)
print('Accuracy on train data set with L2 is : ',accuracy_on_train_L2 )

#E_val_test(model3)
predicted_label_on_val_test_L2 = loaded_model_L2_Unbal_wall.predict(X_val_te_50_arr)
accuracy_on_val_test_L2 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_L2)
print('Accuracy on validation test data set with L2 is : ',accuracy_on_val_test_L2 )

print('Selected parameters : \n' , loaded_model_L2_Unbal_wall.get_params()  , '\n')

#E_train_bal
predicted_label_on_train_bal_L2 = loaded_model_L2_bal_wall.predict(X_tr_bal_arr)
accuracy_on_train_bal_L2 = accuracy_score(y_tr_bal_arr, predicted_label_on_train_bal_L2)
print('Accuracy on balanced train data set with L2 is : ',accuracy_on_train_bal_L2 )

#E_val_test(model4)
predicted_label_on_val_test_bal_L2 = loaded_model_L2_bal_wall.predict(X_val_te_50_arr)
accuracy_on_val_test_bal_L2 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_bal_L2)
print('Accuracy on validation test data set with L2 trained by balanced D'' is : ',accuracy_on_val_test_bal_L2 )

print('Selected parameters : \n' , loaded_model_L2_bal_wall.get_params()  , '\n')


##Chosen L1 models
print('********************** L1 Models************************ \n')
#E_train
predicted_label_on_train_L1 = loaded_model_L1_Unbal_wall.predict(X_tr_80_arr)
accuracy_on_train_L1 = accuracy_score(y_tr_80_arr, predicted_label_on_train_L1)
print('Accuracy on train data set with L1 is : ',accuracy_on_train_L1 )

#E_val_test(model5)
predicted_label_on_val_test_L1 = loaded_model_L1_Unbal_wall.predict(X_val_te_50_arr)
accuracy_on_val_test_L1 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_L1)
print('Accuracy on validation test data set with L1 is : ',accuracy_on_val_test_L1 )

print('Selected parameters : \n' , loaded_model_L1_Unbal_wall.get_params()  , '\n')

#E_train_bal
predicted_label_on_train_bal_L1 = loaded_model_L1_bal_wall.predict(X_tr_bal_arr)
accuracy_on_train_bal_L1 = accuracy_score(y_tr_bal_arr, predicted_label_on_train_bal_L1)
print('Accuracy on balanced train data set with L1 is : ',accuracy_on_train_bal_L1 )

#E_val_test(model6)
predicted_label_on_val_test_bal_L1 = loaded_model_L1_bal_wall.predict(X_val_te_50_arr)
accuracy_on_val_test_bal_L1 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_bal_L1)
print('Accuracy on validation test data set with L1 trained by balanced D'' is : ',accuracy_on_val_test_bal_L1 )

print('Selected parameters : \n' , loaded_model_L1_bal_wall.get_params()  , '\n')


#Final model selection and its final performance measures on D_fin_test
print('********************** Final Model************************ \n')
#Random Forest trained by balanced data set is chosen as final model
#Calculate E_fin_test
#model2
predicted_label_on_fin_test_bal_RF = loaded_model_RF_bal_wall.predict(X_fin_te_50_arr)
accuracy_on_fin_test_bal_RF = accuracy_score(y_fin_te_50_arr, predicted_label_on_fin_test_bal_RF)
print('Accuracy on final test data set with RF trained by balanced D'' is : ',accuracy_on_fin_test_bal_RF )

#Calculate Confusion Matrix
print('Confusion matrix: \n', confusion_matrix(y_fin_te_50_arr, predicted_label_on_fin_test_bal_RF))

#Calculate F-1 Score
print('F1 score: ', f1_score(y_fin_te_50_arr, predicted_label_on_fin_test_bal_RF, average='macro'))

#Final model's parameters
print(print('Selected parameters : \n' , loaded_model_RF_bal_wall.get_params()  , '\n'))


# In[ ]:





# In[ ]:




