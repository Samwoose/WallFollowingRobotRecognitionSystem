#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
def averageCalculator(accuracy_list):
    average = sum(accuracy_list) / len(accuracy_list)
    return average


def logisticRegressionExecuter(X_tr, y_tr, X_te, y_te, regularizer):
    '''
    Input:
    1.X_tr & y_tr : training data set(i.e., features and labels respectively)  , type: data frame or numpy array
    2.X_te & y_te : test data set(i.e., features and labels respectively) , type: data frame or numpy array
    3.regularizer : name of regularizer used in this algorithm. e.g., L1 : Lasso regression, L2: Logistic regression , type:String
    
    Output:
    1.final_c : chosen amount of penalty(regularizer) = 1/lambda  , type: float
    2.reularizer : used regularizer , type: string
    3.ave_accuracy_with_chosen_C : average accuracy with the chosen C. It is found by cross validation , type: float  
    4.fin_accuracy_on_train : final accuracy on the train data set , type: float
    5.fin_accuracy_on_test : final accuracy on the test data set , type: float
    6.fin_LR_cla : final model , type: object
    
    '''
    
    #preassignment session
    c_values_50points = np.logspace(-2 , 3 , num = 50)
    # lambda is big(i.e. 10^2) => C = 0.01 inverse proportionally
    # lambda is small(i.e. 10^-3) => C = 1000 inverse proportionally
    
    number_of_fold = 5
    kf = KFold(n_splits = number_of_fold, shuffle = True)

    #storage for average accuracy for each lambda
    stor_ave_acc = []
    
    #Cross Validation
    for i in range(0, len(c_values_50points)):
        temp_c = c_values_50points[i]
        
        #create a storage for 5 accuracies for same lambda and different train and test set from the whole train set
        temp_acc_list = []
        
        print('Cross Validation Processing : current value of C is ', temp_c )
        
        #calculate accuracy 5 times with the same lambda
        for k in range(0,5):
            
            for train_index , test_index in kf.split(X_tr):
                X_val_tr , X_val_test = X_tr[train_index] , X_tr[test_index]
                y_val_tr , y_val_test = y_tr[train_index] , y_tr[test_index]
                
                X_val_tr_arr = np.array(X_val_tr)
                X_val_test_arr = np.array(X_val_test)
                
                y_val_tr_arr = np.array(y_val_tr)
                y_val_test_arr = np.array(y_val_test)
                
                if(regularizer == 'l1'):
                    #Logistic Regression for each lambda and each time with different train and test set from the whole train set
                    LR_cla_val = LogisticRegression(penalty = regularizer, C = temp_c, solver = 'saga' , max_iter = 1000, multi_class = 'ovr' ).fit(X_val_tr_arr, np.ravel(y_val_tr_arr , order = 'C') )
                
                elif(regularizer == 'l2'):
                    #Logistic Regression for each lambda and each time with different train and test set from the whole train set
                    LR_cla_val = LogisticRegression(penalty = regularizer, C = temp_c, solver = 'sag' , max_iter = 1000 , multi_class = 'ovr' ).fit(X_val_tr_arr, np.ravel(y_val_tr_arr , order='C') )
                
                predicted_label_on_test_data = LR_cla_val.predict(X_val_test_arr)
                
                #calculate accuracy    
                accuracy_on_x_test = accuracy_score(y_val_test_arr, predicted_label_on_test_data)
                
                #save accuracy for each lambda 5 times total
                temp_acc_list.append(accuracy_on_x_test)
                
        #save average of 5 accuracies for each lambda
        average_acc = averageCalculator(temp_acc_list)
        stor_ave_acc.append(average_acc)
    
    #find location of C value
    c_location = stor_ave_acc.index(max(stor_ave_acc))
    final_c = c_values_50points[c_location]
    
    print('chosen C is ' , final_c , '\n')
    print('Average Acurracy with the chosen C is ' , max(stor_ave_acc) , '\n')
    
    #Thanks to code from https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected to resolve data conversion warning 
    #get final logistic regression model using the final c value we found by cross validation
    
    if(regularizer == 'l1'):
        fin_LR_cla = LogisticRegression(penalty = regularizer, C = final_c, solver = 'saga' , max_iter = 1000, multi_class = 'ovr' ).fit(X_tr, np.ravel(y_tr , order = 'C') )
        
    elif(regularizer == 'l2'):
        fin_LR_cla = LogisticRegression(penalty = regularizer, C = final_c, solver = 'sag', max_iter = 1000, multi_class = 'ovr').fit(X_tr, np.ravel(y_tr , order = 'C'))
        
    fin_predicted_label_on_train_data = fin_LR_cla.predict(X_tr)
    
    #calculate accuracy on whole train data set
    fin_accuracy_on_train = accuracy_score(y_tr, fin_predicted_label_on_train_data)
    print('Accuracy on whole train data with chosen C value is : ', fin_accuracy_on_train)
    
    #calculate accuracy on whole test data set
    fin_predicted_label_on_test_data = fin_LR_cla.predict(X_te)
    fin_accuracy_on_test = accuracy_score(y_te, fin_predicted_label_on_test_data)
    print('Accuracy on whole test data with chosen C value is : ', fin_accuracy_on_test)
    
    
    
    return final_c, max(stor_ave_acc) , fin_accuracy_on_train , fin_accuracy_on_test , regularizer , fin_LR_cla
    
    
def randomForestExecuter(X_tr, y_tr, X_te, y_te):
    '''
    Input:
    1.X_tr & y_tr : training data set(i.e., features and labels respectively)  , type: data frame or numpy array
    2.X_te & y_te : test data set(i.e., features and labels respectively) , type: data frame or numpy array
    
    Output:
    1.final_num_estimators : chosen number of estimators  , type: integer
    2.ave_accuracy_with_chosen_num_estimators : average accuracy with the number of the estimators. It is found by cross validation , type: float  
    3.fin_accuracy_on_train_RF : final accuracy on the train data set , type: float
    4.fin_accuracy_on_test_RF : final accuracy on the test data set , type: float
    5.fin_RF_clf : a chosen model after cross validation , type: object
    
    '''
    
    '''
    RF = Random Forest
    '''
    
    
    
    #create possible number of estimators that will be used in cross validation to find best number of estimators
    n_estimators_values_list = []
    for i_RF in range(1,101):
        n_estimators_values_list.append(i_RF*10)
    n_estimators_values_arr = np.array(n_estimators_values_list) #range of number of estimators : from 10 to 1000, step size 10 
    
    num_of_fold_RF = 5
    kf_RF = KFold(n_splits = num_of_fold_RF, shuffle = True)
    
    #storage for average accuracy for each number of estimator
    stor_ave_acc_RF = []
    
    #Cross-Validation
    for j_RF in range(0,len(n_estimators_values_arr)):
        temp_n_estimator = n_estimators_values_arr[j_RF]
        
        #create a storage for 5 accuracies for same number of estimators and different train and test set from the whole train set
        temp_acc_list_RF = []
        
        print('Cross Validation Processing : current number of estimators is ', temp_n_estimator )
        
        for k_RF in range(0,5):
            
            for train_index_RF , test_index_RF in kf_RF.split(X_tr):
                X_val_tr_RF , X_val_test_RF = X_tr[train_index_RF] , X_tr[test_index_RF]
                y_var_tr_RF , y_val_test_RF = y_tr[train_index_RF] , y_tr[test_index_RF]
                
                X_val_tr_RF_arr = np.array(X_val_tr_RF)
                X_val_test_RF_arr = np.array(X_val_test_RF)
                
                y_val_tr_RF_arr = np.array(y_var_tr_RF)
                y_val_test_RF_arr = np.array(y_val_test_RF)
                
                #create D_bag with bag size 1 
                X_train_bag  = X_val_tr_RF_arr
                y_train_bag  = y_val_tr_RF_arr
                #Random Forest Process
                temp_RF_clf = RandomForestClassifier(n_estimators = temp_n_estimator, bootstrap=True)
                #training and predict on test validation set
                temp_RF_clf.fit(X_train_bag, np.ravel(y_train_bag,order='C'))
                predicted_label_on_test_val_data_RF = temp_RF_clf.predict(X_val_test_RF_arr)
                
                #evaluate accuracy on test data set
                accuracy_on_test_val_data_RF = accuracy_score(y_val_test_RF_arr, predicted_label_on_test_val_data_RF)
                
                #save accuracy for each number of estimators 5 times total
                temp_acc_list_RF.append(accuracy_on_test_val_data_RF)
                
        #save average of 5 accuracies for each number of estimators
        average_acc_RF = averageCalculator(temp_acc_list_RF)
        stor_ave_acc_RF.append(average_acc_RF)
    
    #find location of number of estimator value
    num_estimator_location = stor_ave_acc_RF.index(max(stor_ave_acc_RF))
    final_num_of_estimators = n_estimators_values_arr[num_estimator_location]
    print('Chosen number of estimators is ' , final_num_of_estimators , '\n' )
    print('Average Acurracy with the chosen number of estimators is ' , max(stor_ave_acc_RF) , '\n')
    
    #Final training
    fin_RF_clf = RandomForestClassifier(n_estimators = final_num_of_estimators, bootstrap = True)
    fin_RF_clf.fit(X_tr,np.ravel(y_tr,order='C'))
    fin_predicted_label_on_train_data_RF = fin_RF_clf.predict(X_tr)
    
    #calculate accuracy on whole train data set
    fin_accuracy_on_train_RF = accuracy_score(y_tr, fin_predicted_label_on_train_data_RF)
    print('Accuracy on whole train data with chosen number of estimator is : ', fin_accuracy_on_train_RF)
    
    #calculate accuracy on whole test data set
    fin_predicted_label_on_test_data_RF = fin_RF_clf.predict(X_te)
    fin_accuracy_on_test_RF = accuracy_score(y_te, fin_predicted_label_on_test_data_RF)
    print('Accuracy on whole test data with chosen number of estimator is : ' , fin_accuracy_on_test_RF)
    
    
    
    return final_num_of_estimators, max(stor_ave_acc_RF) , fin_accuracy_on_train_RF , fin_accuracy_on_test_RF , fin_RF_clf
            


################### Main part #########################
#Load Preprocessed Datasets
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




####################Logistic Regression(L2 norm, Function version) on Unbalanced Data ###################(less than 16 mins)
c_L2, ave_acc_L2, fin_acc_tr_L2, fin_acc_test_L2, name_reg , fin_LR_cla_L2_unbal= logisticRegressionExecuter(X_tr_80_arr, y_tr_80_arr, X_val_te_50_arr, y_val_te_50_arr, 'l2')

#thanks to code from https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
#How to save finalized model


#save the model to disk
filename_L2_Unbal = 'fin_LR_cla_L2_unbal_model_wall.sav'
pickle.dump(fin_LR_cla_L2_unbal, open(filename_L2_Unbal, 'wb'))


####################Lasso Regression(L1 norm, Function version) on Unbalanced Data ###################(less than 48 mins)
c_L1, ave_acc_L1, fin_acc_tr_L1, fin_acc_test_L1, name_reg , fin_LR_cla_L1_unbal = logisticRegressionExecuter(X_tr_80_arr, y_tr_80_arr, X_val_te_50_arr, y_val_te_50_arr, 'l1')

#save the model to disk
filename_L1_Unbal = 'fin_LR_cla_L1_unbal_model_wall.sav'
pickle.dump(fin_LR_cla_L1_unbal, open(filename_L1_Unbal, 'wb'))


####################Random Forest(function version ) on Unbalanced Data ###################(Less than 3 hours 5 mins)
fin_num_of_estimators, ave_acc_RF , fin_acc_on_tr_RF , fin_acc_on_test_RF , fin_RF_clf_unbal = randomForestExecuter(X_tr_80_arr, y_tr_80_arr, X_val_te_50_arr, y_val_te_50_arr)

#save the model to disk
filename_RF_Unbal = 'fin_RF_clf_unbal_model_wall.sav'
pickle.dump(fin_RF_clf_unbal, open(filename_RF_Unbal, 'wb'))


####################Logistic Regression(L2 norm, Function version) training on balanced Data ###################(less than 11 mins)
c_L2_bal, ave_acc_L2_bal, fin_acc_tr_L2_bal, fin_acc_test_L2_bal, name_reg_L2_bal , fin_LR_cla_L2_bal= logisticRegressionExecuter(X_tr_bal_arr, y_tr_bal_arr, X_val_te_50_arr, y_val_te_50_arr, 'l2')

#save the model to disk
filename_L2_bal = 'fin_LR_cla_L2_bal_model_wall.sav'
pickle.dump(fin_LR_cla_L2_bal, open(filename_L2_bal, 'wb')) 



####################Lasso Regression(L1 norm, Function version) training on balanced Data ###################(less than  60 mins)
c_L1_bal, ave_acc_L1_bal, fin_acc_tr_L1_bal, fin_acc_test_L1_bal, name_reg_L1_bal , fin_LR_cla_L1_bal= logisticRegressionExecuter(X_tr_bal_arr, y_tr_bal_arr, X_val_te_50_arr, y_val_te_50_arr, 'l1')

#save the model to disk
filename_L1_bal = 'fin_LR_cla_L1_bal_model_wall.sav'
pickle.dump(fin_LR_cla_L1_bal, open(filename_L1_bal, 'wb')) 


####################Random Forest(function version ) on balanced Data ###################(Less than 29 mins)
fin_num_of_estimators_bal, ave_acc_RF_bal , fin_acc_on_tr_RF_bal , fin_acc_on_test_RF_bal , fin_RF_clf_bal = randomForestExecuter(X_tr_bal_arr, y_tr_bal_arr, X_val_te_50_arr, y_val_te_50_arr)

#save the model to disk
filename_RF_bal = 'fin_RF_clf_bal_model_wall.sav'
pickle.dump(fin_RF_clf_bal, open(filename_RF_bal, 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




