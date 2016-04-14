# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:22:37 2016

@author: Thiru
"""

import pandas as pd,numpy as np
from sklearn import metrics,cross_validation
from sklearn.preprocessing import LabelEncoder,Imputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


###################################################
#              Preprocessing Methods              #
###################################################
"""
Input:
1) <PD DF> df: pandas dataframe of training data
2) <Boolean> impute: if True, do imputation for missing values rather than dropping

Output:
<PD DF> Processed DF
"""
def preprocess(df,impute):
    def dropRedundantColumns(df):
        dropCols = ['orderID','orderDate']
        df=df.drop(dropCols,axis=1)
        return df
        
     #missing values are in productGroup,rrp,voucherID
    def missingValues(df,impute):
        if impute == False:
            return df.dropna()
        else:
            #remove voucherID, impute the rest.
            df = df[pd.notnull(df['voucherID'])]
            imp = Imputer()
            col1 = imp.fit_transform(df['rrp'].reshape(-1,1))
            col2 = imp.fit_transform(df['productGroup'].reshape(-1,1)) #may not make sense
            df['rrp'] = col1 #find proper way
            df['productGroup'] = col2
            return df
            
    #deal with sizeCode being a bitch and 
    # having S,M,L,I,A, and values also. 
    # Ideally, impute the S,M,L to numeric, but whats I,A??
    def fixSizeCode(df):
        #TEMPORARY, JUST DROP INSTEAD. Find better way!
        df = df.replace(['XS','S','M','L','I','A','XL'],np.nan)  
        return df[pd.notnull(df['sizeCode'])]
        
    def oneHotEncode(df):
        #maybe can drop voucherID, articleID and stuff.
        columnsToEncode=['paymentMethod','customerID','articleID','voucherID']
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df
    
    df = dropRedundantColumns(df)
    df = missingValues(df,impute=impute)
    df = fixSizeCode(df)
    df = oneHotEncode(df)
    return df

"""
Input:
1) <PD DF> df: pandas dataframe

Output:
1) <numpy array> dataset: the features of the training set
2) <numpy array> target: the labels of the training set
"""
def splitDatasetTarget(df):
    dataset = df.drop(['returnQuantity'], axis=1).values
    target = df['returnQuantity'].values
    return dataset,target

###################################################
#                   Models                        #
###################################################

##Dont use this for accuracyChecker. Ran 1+ hr and didnt stop.
def xgBoost():
    clf = xgb.XGBClassifier(max_depth = 6,n_estimators=200,nthread=8,seed=1,
                            objective= 'multi:softmax',learning_rate=0.5,subsample=0.9)
    return clf
    
def randomForest():
    clf = RandomForestClassifier(max_depth=8, n_estimators=300,n_jobs=8,random_state=1)
    return clf
    
###################################################
#                 Testing Models                  #
###################################################

def getNameFromModel(clf):
    name = str(type(clf))
    name = name[name.rfind('.')+1:name.rfind("'")] #subset from last . to last '
    return name

"""
Input:
1) <pd df> dataset: Pandas dataframe of features
2) <pf df> target: Pandas 1D dataframe of target labels
3) <List or classifier> clfs: List of classifiers of single classifier

Output:
1) None. Prints 5 fold cross val score, confusion matrix, and competition metric
for all classifiers. 
"""
def accuracyChecker(dataset,target,clfs):
    if type(clfs) != list:
        clfs = [clfs]
    
    for classifier in clfs:
        name= getNameFromModel(classifier)
        print('******** '+name+' ********')
        predicted = cross_validation.cross_val_predict(classifier,dataset,target,cv=5)
        print('5 fold cross val score for '+name+' : '+str(round(metrics.accuracy_score(target,predicted)),2))
        print(metrics.confusion_matrix(target,predicted,labels=[0,1,2,3,4,5]))
        print('Competition metric score : '+str(computeError(predicted,target)))
    
    

"""
Input:
1) <PD DF> predicted: pandas df of predicted labels
2) <PD DF> target: 1D df/array of target label.

Output:
<Integer> Sum of errors of predicted vs target
"""
def computeError(predicted,target):
    return sum(abs(predicted-target))
    
def run():
    train = pd.read_csv('E:/Git/DMC2016/thirufiles/orders_train.csv',sep=';')
    train = preprocess(train,False)
    dataset,target = splitDatasetTarget(train)
    clfs = [randomForest()]
    accuracyChecker(dataset,target,clfs)
    
    
if __name__ == '__main__':
	run()
