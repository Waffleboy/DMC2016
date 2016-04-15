# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:22:37 2016

@author: Thiru
"""

import pandas as pd,numpy as np
from sklearn import metrics,cross_validation
from sklearn.preprocessing import LabelEncoder,Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit,train_test_split
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
    df.reset_index(inplace=True,drop=True)
    return df

"""
Input:
1) <PD DF> df: pandas dataframe

Output:
1) <numpy array> dataset: the features of the training set
2) <numpy array> target: the labels of the training set
"""
def splitDatasetTarget(df):
    dataset = df.drop(['returnQuantity'], axis=1)
    target = df['returnQuantity']
    return dataset,target
###################################################
#          Stratified Sample Functions            #
###################################################

"""
Input:
 1)<PD DF> dataset: input features
 2)<PD DF> target:  labels
 
Output:

1) <PD DF> Stratified sample of dataset
2) <PD DF> Stratified sample of label
"""
#NOTE: Currently configured to only return ONE sample.
# SELF CODED CAUSE SKLEARN IS A FKING BURDEN
def stratifiedSampleGenerator(dataset,target,subsample_size=0.1):
    X_fit,X_eval,y_fit,y_eval= train_test_split(dataset,target,test_size=subsample_size,stratify=target)
    return X_eval.reset_index(drop=True),y_eval.reset_index(drop=True)
#    print('Generating stratified sample of size ' + str(round(len(dataset)*subsample_size,2)))
#    dic={}
#    indexes = np.array([])
#    # find number of classes in sample
#    for label in target.unique(): 
#         labelSize = len(target[target==label]) 
#         dic[label] = int(labelSize * subsample_size)
#    # make a dataset of size sizeSample with ratio of classes in dic
#    for label in dic:
#        classIndex = target[target==label].index #obtain indexes of class
#        counts = dic[label]   #get number of times class occurs
#        newIndex = np.random.choice(classIndex,counts,replace=False)
#        indexes = np.concatenate((indexes,newIndex),axis=0)
#        
#    indexes = indexes.astype(int)
#    sampleData = dataset.ix[indexes]
#    sampleData.drop('index',inplace=True,axis=1) #cause ix generates indexes as a column ffs
#    sampleTarget = target[indexes]
#    sampleData['returnQuantity'] = sampleTarget
#    sampleData.reset_index(inplace=True)
#    sampleData = sampleData.iloc[np.random.permutation(len(sampleData))] #shuffle dataset
#    sampleData.drop('index',inplace=True,axis=1)
#    sampleData.reset_index(inplace=True)
#    
#    newDataset = sampleData.drop('returnQuantity',axis=1)
#    newTarget = sampleData['returnQuantity']
#    if newDataset['sizeCode'].ftype != 'int32:dense':
#        newDataset['sizeCode'] = newDataset['sizeCode'].astype(int)
#        
#    return newDataset,newTarget
            
        
###################################################
#                   Models                        #
###################################################

##Dont use this for accuracyChecker. Ran 1+ hr and didnt stop.
def xgBoost():
    clf = xgb.XGBClassifier(max_depth = 6,n_estimators=200,nthread=8,seed=1,silent=1,
                            objective= 'multi:softmax',learning_rate=0.03,subsample=0.9)
    return clf
    
def randomForest():
    clf = RandomForestClassifier(max_depth=8, n_estimators=200,n_jobs=8,random_state=1)
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
    print('Beginning evaluation of models...')
    if type(clfs) != list:
        clfs = [clfs]
    
    for classifier in clfs:
        #check if xgboost. if so, pass to XGBChecker. else, continue normally.
        name= getNameFromModel(classifier)
        if name == 'XGBClassifier':
            XGBChecker(dataset,target,classifier)
            continue
        print('******** '+name+' ********')
        predicted = cross_validation.cross_val_predict(classifier,dataset,target,cv=5)
        print('5 fold cross val accuracy for '+name+': '+str( round(metrics.accuracy_score(target,predicted)*100,2) )+'%')
        print(metrics.confusion_matrix(target,predicted,labels=[0,1,2,3,4,5]))
        print(name + 'Competition metric score : '+str(computeError(predicted,target)))

"""
Function made to specifically handle xgboost. works similar to accuracyChecker.
Set to 5 fold CV. 
"""    
def XGBChecker(dataset,target,classifier):
    print('******** XGBClassifier ********')
    cvList = []
    predicted = np.array([])
    fold = 0
    sss = StratifiedShuffleSplit(target,5,test_size=0.2, random_state=0)    
    for train_index, test_index in sss:
        fold+=1
        print('Training Xgboost fold '+str(fold))
        trainX = dataset.ix[train_index].drop('index',axis=1).values # trainX[train_index] doesnt work tho it should
        trainY = target[train_index].values
        testX = dataset.ix[test_index].drop('index',axis=1).values
        testY = target[test_index].values
        
        print(len(trainX),len(trainY),len(testX),len(testY))
        
        classifier.fit(trainX,trainY, early_stopping_rounds=25, 
                       eval_metric="merror", eval_set=[(testX, testY)])
                       
        pred = classifier.predict(testX)
        predicted = np.concatenate([predicted,pred])
        cvList.append(metrics.accuracy_score(testY,pred))
    
    print('Xgboost 5 fold cv: '+str(cvList))
    print('Average CV Score: '+ str(np.mean(cvList)))
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
    print('Processed data. Splitting..')
    dataset,target = splitDatasetTarget(train)
    dataset,target = stratifiedSampleGenerator(dataset,target,subsample_size=0.1)
    clfs = [randomForest()]
    accuracyChecker(dataset,target,clfs)
    
    
if __name__ == '__main__':
	run()
