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
from scipy import stats
# from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# If existing processed csv exists, load it. Else, load raw dataset and run
# preprocessing and feature engineering
def loadDataFrame():
    check = True
    if os.path.exists('preprocessed.csv'):
        print("Loading feature engineered dataset")
        df = pd.read_csv('preprocessed.csv')
        check = False
    else:
        print("Loading original dataset")
        COM_NAME = socket.gethostname()
        if COM_NAME == 'Waff1e':
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
            dropCols = ['orderID','quantity']
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
    df.drop('voucherAmount',inplace=True,axis=1) #temporary hack
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
        return df.dropna() #fix missing values. Else first run cannot run

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
        df.drop('totalSpent',axis=1,inplace=True)
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
        #df['totalPurchases']=totalPurchases                   #decreases accuracy
        #df['purchaseFrequency'] = totalPurchases / numMonths  #decreases accuracy
        return df
    
    #Creates 4 columns
    # 1) modeSize: most frequent size bought by customer
    # 2) differenceModeSize: difference between modeSize and specific item bought
    # 3) averageSize: average(mean) of the size bought by customer
    # 4) differenceAvgSize: difference between averageSize and specific item bought
    def modeSize(df):
        print('Making: mostFrequentSize and differenceSize')
        if not os.path.exists('pickleFiles/modeSizesBought.pkl') and not os.path.exists('pickleFiles/averageSize.pkl'):
            allSize = {}
            for i in df.index: #find all sizes purchased by customers
                currCust = df['customerID'][i]
                if currCust not in size:
                    allSize[currCust] = [df['sizeCode'][i]]
                else:
                    allSize[currCust].append(df['sizeCode'][i])
            modeSize = {}
            averageSizeData={}
            #fill up modeSize and averageSizeData
            for customer in allSize:
                if customer not in modeSize:
                    mode = Counter(allSize[customer]).most_common(1)[0][0]
                    modeSize[customer] = mode
                    avg = np.mean(allSize[customer])
                    averageSizeData[customer] = avg
        else:
            modeSizeData = joblib.load('pickleFiles/modeSizesBought.pkl')
            averageSizeData = joblib.load('pickleFiles/averageSize.pkl')
            
        mostFrequentSize = pd.Series(name= 'mostFrequentSize', index=df.index)
        averageSize = pd.Series(name= 'averageSize', index=df.index)
        
        for i in df.index:
            customer = df['customerID'][i]
            mostFrequentSize.set_value(i,modeSizeData[customer])
            averageSize.set_value(i,averageSizeData[customer])
        
        df['modeSize'] = mostFrequentSize
        df['differenceModeSize'] = abs(mostFrequentSize - df['sizeCode'])
        df['averageSize'] = averageSize
        df['differenceAvgSize'] = abs(averageSize - df['sizeCode'])
        return df
    
    #Creates 2 columns
    # 1) averageColor: the average colorCode that each customer buys
    # 2) differenceAvgColor: the difference between the specific item bought and averageColor
    def averageColor(df):
        if not os.path.exists('pickleFiles/averageColor.pkl'):
            allColor = {} #find all the colours that customers buy
            for i in df.index:
                currCustomer = df['customerID'][i]
                if currCustomer not in allColor:
                    allColor[currCustomer] = [df['colorCode'][i]]
                else:
                    allColor[currCustomer].append(df['colorCode'][i])
            averageColor = {}
            for entry in allColor:
                if entry not in averageColor:
                    averageColor[entry] = np.mean(allColor[entry])
        else:
            averageColor = joblib.load('pickleFiles/averageColor.pkl')
        avgcolor = pd.Series(name= 'averageColor', index=df.index)
        for i in df.index:
            customer = df['customerID'][i]
            avgcolor.set_value(i,averageColor[customer])
        df['averageColor'] = avgcolor
        df['differenceAvgColor'] = avgcolor - df['colorCode']
        return df

    """
    Creates two columns: popular colors and size for each articleID.
    """
    def articlePopularity(df):
        if not os.path.exists('pickleFiles/popularSizeByArticle.pkl') and not os.path.exists('pickleFiles/popularColorByArticle.pkl'):
            popSizeDic, popColorDic = {}, {}
            articles = df.groupby('articleID')
            for idx,article in articles:
                if idx not in popSizeDic or idx not in popColorDic:
                    popColorDic[idx] = Counter(article['colorCode']).most_common()[0][0]
                    popSizeDic[idx] = Counter(article['sizeCode']).most_common()[0][0]
        else:
            popColorDic = joblib.load('pickleFiles/popularColorByArticle.pkl')
            popSizeDic = joblib.load('pickleFiles/popularSizeByArticle.pkl')
        popColor = pd.Series(name='popularColorByArticle', index=df.index)
        popSize = pd.Series(name='popularSizeByArticle', index=df.index)
        for i in df.index:
            article = df['articleID'][i]
            if article in popColorDic:
                colorCheck = 1 if df['colorCode'][i] == popColorDic[article] else 0
            else:
                colorCheck = 0
            if article in popSizeDic:
                sizeCheck = 1 if df['sizeCode'][i] == popSizeDic[article] else 0
            else:
                sizeCheck = 0
            popColor.set_value(i,colorCheck)
            popSize.set_value(i,sizeCheck)
        df['popularColorByArticle'] = popColor
        df['popularSizeByArticle'] = popSize
        return df

    """
    Generates a boolean column indicating if an article tends to be purchased moreso when a voucher is used
    """
    def cheapskateItems(df):
        if not os.path.exists('pickleFiles/voucherToArticle.pkl'):
            voucherDic = {}
            vouchers = df.groupby('voucherID')
            for idx,voucher in vouchers:
                if idx not in voucherDic:
                    voucherDic[idx] = Counter(voucher['articleID']).most_common()[0][0]
        else:
            voucherDic = joblib.load('pickleFiles/voucherToArticle.pkl')
        articleSet = set(voucherDic.values())
        cheapArticle = pd.Series(name='cheapArticle',index=df.index)
        for i in df.index:
            article = df['articleID'][i]
            isCheap = 1 if article in articleSet else 0
            cheapArticle.set_value(i,isCheap)
        df['cheapArticle'] = cheapArticle
        return df

    """
    Generates the standard deviation amongst counts of each product group
    """
    def varianceInProductGroups(df):
        if not os.path.exists('pickleFiles/colorStd.pkl') and not os.path.exists('pickleFiles/sizeStd.pkl'):
            products = df.groupby('productGroup')
            sizeStd, colorStd = {},{}
            for idx,product in products:
                if idx not in sizeStd or idx not in colorStd:
                    size = np.std(list(Counter(product['sizeCode']).values()))
                    color = np.std(list(Counter(product['colorCode']).values()))
                    sizeStd[idx] = size
                    colorStd[idx] = color
        else:
            sizeStd = joblib.load('pickleFiles/sizeStd.pkl')
            colorStd = joblib.load('pickleFiles/colorStd.pkl')
        df['sizeStd'] = df['productGroup'].map(sizeStd)
        df['colorStd'] = df['productGroup'].map(colorStd)
        return df

    def isRepeatCustomer(df):
        d = {}
        for i in df.index:
            idx = df['customerID']
            if idx not in d:
                d[idx] = 1
            else:
                d[idx] += 1
        singlePurchase = [key for key in d if d[key]==1]
        for key in d:
            if d[key] == 1:
                singlePurchase.append(key)
        repeatCustomer = pd.Series(name='repeatCustomer',index=df.index)
        for j in df.index:
            isRepeat = 1 if df['customerID'][j] in repeatCustomer else 0
            repeatCustomer.set_value(j,isRepeat)
        df['repeatCustomer'] = repeatCustomer
        return df

    def weekendWeekday(df):
        return df

    def multiplePurchase(df):
        return df

    def paymentType(df):
        return df

    def highReturnItem(df):
        return df
        
    # 1) repeatcustomer column. Yes or no
    # 2) whether they ordered on a weekend or weekday. 2 columns both boolean 
    # 3) did they buy more than one item. Yes or no
    # 4) online or offline payment? 
    # 5) more tricky. Whether that individual item is a high return item

    df = purchasesAndReturns(df)
    df = userSpending(df) 
    df = priceDiscount(df)
    df = colorPopularity(df)
    df = modeSize(df)
    df = averageColor(df)
    df = articlePopularity(df)
    df = cheapskateItems(df)
    df = varianceInProductGroups(df)
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
    # dataset = df.drop(['returnQuantity'], axis=1)
    dataset = df.drop(['returnQuantity','popularSizeByArticle','cheapArticle','sizeStd','colorStd','averageColor','differenceAvgColor','modeSize','differenceModeSize','averageSize','differenceAvgSize'], axis=1)
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
    clf = xgb.XGBClassifier(max_depth = 8,n_estimators=300,nthread=8,seed=1,silent=1,
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

#Iterates through dataset, drops a column and fits classifier to find change in accuracy
def testFeatureAccuracy(dataset,target):
    lst=[['Without Feature','Total Accuracy','Net Change']]
    
    #find base accuracy first
    trainx,testx,trainy,testy = train_test_split(dataset,target,test_size=0.2)
    classifier = xgBoost()
    classifier.fit(trainx,trainy, early_stopping_rounds=25, 
                       eval_metric="merror", eval_set=[(testx, testy)])
    score = classifier.score(testx,testy)
    lst.append(['Baseline',score,'-'])
    initialAccuracy = score
    #for every column in columnsToTest, drop that column then fit
    columnsToTest = ['popularColorByArticle','popularSizeByArticle','cheapArticle','sizeStd','colorStd'] #fill in with column names
    for col in columnsToTest: 
        trainx2 = trainx.drop(col,axis=1)
        testx2 = testx.drop(col,axis=1)
        classifier.fit(trainx2,trainy, early_stopping_rounds=25, 
                       eval_metric="merror", eval_set=[(testx2, testy)])
        score = classifier.score(testx2,testy)
        lst.append([col,score,round(initialAccuracy-score,3)]) #attach score
    df = pd.DataFrame(lst)
    df.to_csv('testAccuracy2.csv',index=False)
    return lst
        
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
        trainx,testx,trainy,testy = train_test_split(dataset,target,test_size=0.2) #70 - 20 split
    predictions = [] 
    # function to show error wrt sample size of data
    def errorScaler(error):
        global datasetSize
        return (error*datasetSize) / len(testx)
    
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
    # testFeatureAccuracy(dataset,target)
    # clfs = [xgBoost(),randomForest(),extraTrees(),kNN(),neuralNetwork()]
    clfs = [xgBoost(),randomForest(),extraTrees(),kNN()]
    clfs = accuracyChecker(dataset,target,clfs,cross_val=False,ensemble = True,record = True,predictTest=False) # Dont use CV, Yes ensemble, Yes Record. 
    
if __name__ == '__main__':
	run()
