# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:22:37 2016

@author: Thiru
"""
import os,csv,socket
import pandas as pd,numpy as np
from sklearn import metrics,cross_validation
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder,Imputer
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.cross_validation import StratifiedShuffleSplit,train_test_split
from collections import Counter
# from sklearn.neural_network import MLPClassifier
import xgboost as xgb

#INTERESTING: For products where sizeCode was ALPHABETICAL and colorCode > 5000: over 50% of them were returned!

def loadDataFrame():
    check = True
    if os.path.exists('preprocessed.csv'):
        print("Loading feature engineered dataset")
        df = pd.read_csv('preprocessed.csv')
        check = False
    else:
        print("Loading original dataset")
        COM_NAME = socket.gethostname()
        if COM_NAME == 'Waffle':
            df = pd.read_csv('E:/Git/DMC2016/thirufiles/orders_train.csv',sep=';')
        else:
            df = pd.read_csv('/home/andre/workshop/dmc2016/andrefiles/orders_train.csv',sep=';')
        df = preprocess(df,impute=False,engineerFeatures=check) #False = dont use imputation.
    return df

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
def preprocess(df,impute,engineerFeatures):
    def dropRedundantColumns(df):
        try:
            dropCols = ['orderID']
            df=df.drop(dropCols,axis=1)
        except:
            pass
        return df
        
    #missing values are in productGroup,rrp,voucherID
    def missingValues(df,impute):
        if impute == False:
            return df.dropna()
        else:
            try:
            #remove voucherID, impute the rest.
                df = df[pd.notnull(df['voucherID'])]
                imp = Imputer()
                col1 = imp.fit_transform(df['rrp'].reshape(-1,1))
                col2 = imp.fit_transform(df['productGroup'].reshape(-1,1)) #may not make sense
                df['rrp'] = col1 #find proper way
                df['productGroup'] = col2
            except:
                print('Error with Imputation')
            return df
        
    #deal with sizeCode being a bitch and 
    # having S,M,L,I,A, and values also. 
    # Ideally, impute the S,M,L to numeric, but whats I,A??
    def fixSizeCode(df):
        #TEMPORARY, CONVERT TO NUMERIC. Find better way!
        df = df.replace(['XS'],200)  
        df = df.replace(['S'],210)  
        df = df.replace(['M'],220)  
        df = df.replace(['L'],230)  
        df = df.replace(['XL'],240)  
        df = df.replace(['I'],250) 
        df = df.replace(['A'],260)  
        df.sizeCode = df.sizeCode.astype(np.int64)
        return df
        
    """
    Convert orderDate to months (12/1/2016 --> 1)
    """
    def orderDateToMonths(df):
        df['orderDate']= pd.DatetimeIndex(pd.to_datetime(df['orderDate'])).month
        return df
    """
    one shot find all categorical columns and encode them. Cause its awesome like that.
    """    
    def oneHotEncode(df):
        #maybe can drop voucherID, articleID and stuff.
     #  columnsToEncode=['paymentMethod','customerID','articleID','voucherID']
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df
    
    print('Dropping redundant columns..')
    df = dropRedundantColumns(df)
    print('Dropping missing values: impute = '+str(impute))
    df = missingValues(df,impute=impute)
    print('Replacing sizeCode..')
    df = fixSizeCode(df)
    print('Changing orderDate to months..')
    df=orderDateToMonths(df)
    df.reset_index(inplace=True,drop=True)
    if engineerFeatures:
        print('Running feature engineering..')
        df = featureEngineering(df)
    print('Encoding all categorical and object vars to numeric')
    df = oneHotEncode(df)
    print('Processing done. Saving CSV')
    df.reset_index(inplace=True,drop=True)
    df.to_csv('preprocessed.csv',index=False)
    return df
    
###################################################
#           Feature Engineering Methods           #
###################################################
    
def featureEngineering(df):
    """
    Returns a column describing how much of the original price was waived by the voucher
    """
    def priceDiscount(df):
        print('Making: priceDiscount')
        priceDiscount = df['voucherAmount'].divide(df['price'],fill_value=0.0)
        priceDiscount[np.isinf(priceDiscount)] = 0.0
        df['priceDiscount'] = priceDiscount
        return df

    """
    Creates new column to indicate if a color is popular or not.
    """
    def colorPopularity(df):
        print('Making: colorPopularity')
        if not os.path.exists('pickleFiles/colorMap.pkl'):
            colorCount = Counter(df['colorCode'])
            popularColors = [i[0] for i in colorCount.most_common(5)]
            shittyColors = [j[0] for j in colorCount.most_common()[::-1] if j[1] < 5]
            colorMap = {}
            for color in df['colorCode']:
                if color not in colorMap:
                    if color in popularColors:
                        colorMap[color] = "popular"
                    elif color in shittyColors:
                        colorMap[color] = "unpopular"
                    else:
                        colorMap[color] = "neutral"
        else:
            colorMap = joblib.load('pickleFiles/colorMap.pkl')
        df['colorPopularity'] = df['colorCode'].map(colorMap)
        return df

    """
    Add relative price of each transaction with respect to overall mean, mode, and median
    """
    def relativePrice(df):
        print('Making: relativePrice')
        # helper function to create dictionaries
        if not os.path.exists('pickleFiles/meanMap.pkl') and not os.path.exists('pickleFiles/modeMap.pkl') and not os.path.exists('pickleFiles/medianMap.pkl'):
            def updateDict(curr,dic,stat):
                if curr not in dic:
                    if curr > stat:
                        dic[curr] = 1
                    else:
                        dic[curr]=0
            prices = df['price']
            lst = prices.values.tolist()
            mode = max(set(lst),key=lst.count)
            median = np.median(prices)
            mean = np.mean(prices)
            meanMap, modeMap, medianMap = {},{},{}
            for i in df.index:
                curr = df['price'][i]
                updateDict(curr,meanMap,mean)
                updateDict(curr,modeMap,mode)
                updateDict(curr,medianMap,median)
        else:
            meanMap = joblib.load('pickleFiles/meanMap.pkl')
            modeMap = joblib.load('pickleFiles/modeMap.pkl')
            medianMap = joblib.load('pickleFiles/medianMap.pkl')
        
        df['moreThanMean'] = df['price'].map(meanMap)
        df['moreThanMedian'] = df['price'].map(medianMap)
        df['moreThanMode'] = df['price'].map(modeMap)
        return df

    """
    Create totalSpent by customer column as well as averageSpent
    """
    def userSpending(df):
        print('Making: userSpending')
        if not os.path.exists('pickleFiles/totalSpent.pkl') and not os.path.exists('pickleFiles/count.pkl') and not os.path.exists('pickleFiles/averageSpent.pkl'):
            totalSpent,count,averageSpent = {},{},{}
            for i in df.index:
                userId = df['customerID'][i]
                price = df['price'][i]
                if userId not in totalSpent:
                    totalSpent[userId] = price
                    count[userId] = 1
                else:
                    totalSpent[userId] += price
                    count[userId] += 1
        else:
            totalSpent = joblib.load('pickleFiles/totalSpent.pkl')
            count = joblib.load('pickleFiles/count.pkl')
            averageSpent = joblib.load('pickleFiles/averageSpent.pkl')
            
        for i in totalSpent:
            averageSpent[i] = totalSpent[i] / count[i]
        df['totalSpent'] = df['customerID'].map(totalSpent)
        df['averageSpent'] = df['customerID'].map(averageSpent)
        df['yearlyExpense'] = df['averageSpent'] / df['totalSpent']
        return df
        
    """
    Create totalPrice column
    """
    def totalPrice(df): #no point from feature importance graph
        print('Making: totalPrice')
        df['totalPrice'] = df['price']*df['quantity']
        return df
    
    # 2 in 1 function to speed up as same loop.
    # 1) Create returnsPerCustomer column, find the total amount of returns per unique
    # customer.
    # 2) create totalPurchases column
    # 3) create purchaseFrequency column
    def purchasesAndReturns(df): #SLOW.
        print('Making: returnsPerCustomer_totalPurchases')
        returnsPerCustomer = pd.Series(name= 'returnsPerCustomer', index=df.index)
        totalPurchases = pd.Series(name= 'totalPurchases', index=df.index)
        
        data  = joblib.load('pickleFiles/returnsPerCustomer.pkl') #of form: {ID:quantity} eg, {a0123134: 5}
        data2 = joblib.load('pickleFiles/totalPurchasesPerCustomer.pkl') #of form: {ID:quantity} eg, {a0123134: 3}
        
        numMonths = len(df['orderDate'].unique()) #find num months in dataset
        #for each customer in customer ID, lookup data and fill in
        for i in df.index: 
            customer = df['customerID'][i]
            returnsPerCustomer.set_value(i,data[customer]) #
            totalPurchases.set_value(i,data2[customer]) 
        
        def simulateTestData(returnsPerCustomer): # replace 2/3 with missing values.
            import random as rand
            length = len(returnsPerCustomer)
            randomList = rand.sample(range(1,length), int(length*(2/3)))
            returnsPerCustomer[randomList] = -99
            return returnsPerCustomer
        
        returnsPerCustomer = simulateTestData(returnsPerCustomer)
        df['returnsPerCustomer']=returnsPerCustomer
        df['totalPurchases']=totalPurchases
        df['purchaseFrequency'] = totalPurchases / numMonths
        return df
    
    def encodeColorCode(df): #decreases accuracy
        print('Encoding Color Code')
        df['colorCode'] = df['colorCode']//100
        return df
        
    def differenceRRPprice(df): 
        print('Making: differenceRRPprice')
        df['rrp-price'] = df['rrp'] - df['price']
        return df
        
    df = totalPrice(df)
    df = purchasesAndReturns(df)
    df = userSpending(df)
    df = differenceRRPprice(df)
    df = relativePrice(df)
    df=  priceDiscount(df)
    df = colorPopularity(df)
    #df = encodeColorCode(df)
    print('Feature Engineering Done')
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
def stratifiedSampleGenerator(dataset,target,test_size=0.1):
    X_fit,X_eval,y_fit,y_eval= train_test_split(dataset,target,test_size=test_size,stratify=target)
    return X_eval.reset_index(drop=True),y_eval.reset_index(drop=True)

###################################################
#                   Models                        #
###################################################

##Dont use this for accuracyChecker. Ran 1+ hr and didnt stop.
def xgBoost():
    clf = xgb.XGBClassifier(max_depth = 8,n_estimators=250,nthread=8,seed=1,silent=1,
                            objective= 'multi:softmax',learning_rate=0.1,subsample=0.9)
    return clf
    
def randomForest():
    clf = RandomForestClassifier(max_depth=8, n_estimators=300,n_jobs=8,random_state=1,
                                 class_weight={0:2,1:1})
    return clf

def extraTrees():
    clf = ExtraTreesClassifier(max_depth=8, n_estimators=300,n_jobs=8,random_state=1,
                               class_weight = {0:1,1:2})
    return clf
    
def kNN():
    clf = KNeighborsClassifier(n_neighbors=1,n_jobs=8)
    return clf
    
def neuralNetwork():
    clf = MLPClassifier(activation='relu',hidden_layer_sizes = (500,300,6),max_iter=500,
                        random_state=1,early_stopping=True)
    return clf
    
###################################################
#              Optimize Models                    #
###################################################
 
"""
Input: params of form {'parameter':[<range>]}
eg.

{'max_depth':[5,6,7,8], 'subsample':[0.5,0.6]}
"""  
def optimizeClassifier(dataset,target,clf,params):
    gsearch = GridSearchCV(estimator = clf, param_grid = params, 
                           scoring='f1_macro',n_jobs=8,iid=True, cv=5) #write own scorer?
    gsearch.fit(dataset,target)
    print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)
    
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
4) <Boolean> cross_val: if True, use cross validation. else, normal train test split.
5) <Boolean> ensemble: if True, ensemble all the classifiers you used. Else, no
6) <Boolean> record: if True, will record to CSV.

**Currently ensemble only works if cross_val == False.

Output:
1) None. Prints 5 fold cross val score, confusion matrix, and competition metric
for all classifiers. 
"""
def accuracyChecker(dataset,target,clfs,cross_val,ensemble,record,predictTest):
    print('Beginning evaluation of models...')
    #if one classifier, make it into lst so not to break function
    if type(clfs) != list:
        clfs = [clfs] 
    #if not cross val, split now to save memory.
    if cross_val == False:
        trainx,testx,trainy,testy = train_test_split(dataset,target,test_size=0.3) #70 - 30 split
    predictions = [] 
    # function to show error wrt sample size of data
    def errorScaler(error):
        global datasetSize
        return (error*datasetSize) / len(dataset)
    
    ### BEGIN ACTUAL FUNCTIONS ###
    for i in range(len(clfs)):
        classifier = clfs[i]
        #check if xgboost. if so, pass to XGBChecker. else, continue normally.
        name= getNameFromModel(classifier)
        if name == 'XGBClassifier' and cross_val == True:
            XGBChecker(dataset,target,classifier)
            continue
        
        if cross_val == True: #if cross val, do this. else, use train test split.
            print('******** '+name+' ********')
            predicted = cross_validation.cross_val_predict(classifier,dataset,target,cv=5)
            print('5 fold cross val accuracy for '+name+': '+str( round(metrics.accuracy_score(target,predicted)*100,2) )+'%')
            print(metrics.confusion_matrix(target,predicted,labels=[0,1,2,3,4,5]))
            error = computeError(predicted,target)
            print(name + 'Competition metric score : '+str(error))
            print(name + 'Competition metric score adjusted for train size: '+str(errorScaler(error)))
            
        else: #use train test split
            #special fit for xgbclassifier
            if name == 'XGBClassifier':
                classifier.fit(trainx,trainy, early_stopping_rounds=25, 
                       eval_metric="merror", eval_set=[(testx, testy)])
            else:
                classifier.fit(trainx,trainy)
            clfs[i] = classifier # set the fitted classifier to lst
            pred = classifier.predict(testx)
            if ensemble: #if ensemble, append to pred to use later
                predictions.append(pred)
            #calculate metrics
            testAccuracy = classifier.score(testx,testy)
            confMat = metrics.confusion_matrix(testy,pred,labels=[0,1,2,3,4,5])
            error = computeError(pred,testy)
            scaledError = errorScaler(error)
            print('Test data accuracy for '+name+': '+ str(testAccuracy))
            print(confMat)
            print(name + ' Competition metric score : '+str(error))
            print(name + 'Competition metric score adjusted for train size: '+str(scaledError))
            if record:
                params = classifier.get_params()
                dataSize = len(testy)
                writeToCSV(name,params,cross_val,dataSize,testAccuracy,confMat,error,scaledError)
                
     #if ensemble, do ensemble stuff       
    if ensemble and len(clfs) >= 2 and cross_val == False:
        predictions = np.array(predictions) #transpose it
        predictions = predictions.T
        clf = ExtraTreesClassifier(max_depth = 5,n_jobs=8,n_estimators=100)
        predicted = cross_validation.cross_val_predict(clf,predictions,testy,cv=5)
        testAccuracy = round(metrics.accuracy_score(testy,predicted),5)
        confMat = metrics.confusion_matrix(testy,predicted,labels=[0,1,2,3,4,5])
        error = computeError(predicted,testy)
        scaledError= errorScaler(error)
        
        print('5 fold cross val accuracy for ensemble '+str(testAccuracy))
        print(confMat)
        print('Ensemble Competition metric score : '+str(error))
        print('Ensemble Competition metric score adjusted for train size: '+str(scaledError))
        if record:
            params = classifier.get_params()
            dataSize = len(testy)
            writeToCSV('Ensemble',params,True,dataSize,testAccuracy,confMat,error,scaledError)
        if predictTest:
            clf.fit(predictions,testy)
            clfs.append(clf)
    return clfs

#Function to write inputs to CSV        
def writeToCSV(name,params,cross_val,size,testAccuracy,confMat,error,scaledError):
    fileName = 'resultsCSV_Thiru.csv'
    if os.path.isfile(fileName) == False:
        with open(fileName,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Classifier','Params','Cross Val','Sample_Size','Accuracy','conf_Matrix','Competition_Error','Scaled_Error'])
    with open(fileName,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name,params,cross_val,size,testAccuracy,confMat,error,scaledError])

"""
Function made to specifically handle xgboost. works similar to accuracyChecker.
Set to 5 fold CV. 
"""    
def XGBChecker(dataset,target,classifier):
    print('******** XGBClassifier ********')
    cvList = []
    predicted = np.array([])
    newTarget = np.array([]) #hackish solution. 
    fold = 0
    sss = StratifiedShuffleSplit(target,5,test_size=0.2, random_state=0)    
    for train_index, test_index in sss:
        fold+=1
        print('Training Xgboost fold '+str(fold))
        trainX = dataset.iloc[train_index] # trainX[train_index] doesnt work tho it should
        trainY = target[train_index]
        testX = dataset.iloc[test_index]
        testY = target[test_index]
        
        classifier.fit(trainX,trainY, early_stopping_rounds=25, 
                       eval_metric="merror", eval_set=[(testX, testY)])
                       
        pred = classifier.predict(testX)
        predicted = np.concatenate((predicted,pred))
        newTarget = np.concatenate((testY,newTarget))
        cvList.append(metrics.accuracy_score(testY,pred))
    
    print('Xgboost 5 fold cv: '+str(cvList))
    print('Average CV Score: '+ str(np.mean(cvList)) + ' +/- ' + str(np.std(cvList)))
    print(metrics.confusion_matrix(newTarget,predicted,labels=[0,1,2,3,4,5]))
    print('Competition metric score : '+str(computeError(predicted,newTarget)))        

"""
Input:
1) <PD DF> predicted: pandas df of predicted labels
2) <PD DF> target: 1D df/array of target label.

Output:
<Integer> Sum of errors of predicted vs target
"""
def computeError(predicted,target):
    return sum(abs(predicted-target))
    
###################################################
#               Generate Predictions              #
###################################################
    
def generatePredictions(clfs,ensemble):
    pass
    
def run():
    train = loadDataFrame()
    global datasetSize
    datasetSize = len(train)
    dataset,target = splitDatasetTarget(train)
    dataset,target = stratifiedSampleGenerator(dataset,target,test_size=0.2)
    # clfs = [xgBoost(),randomForest(),extraTrees(),kNN(),neuralNetwork()]
    clfs = [xgBoost(),randomForest(),extraTrees(),kNN()]
    clfs = accuracyChecker(dataset,target,clfs,cross_val=False,ensemble = True,record = True,predictTest=False) # Dont use CV, Yes ensemble, Yes Record. 
    
if __name__ == '__main__':
	run()
