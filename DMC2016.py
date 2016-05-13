# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:22:37 2016

@author: Thiru
"""
import os,csv,socket,datetime
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
    if os.path.exists('preprocessed_train.csv'):
        print("Loading feature engineered dataset")
        df = pd.read_csv('preprocessed_train.csv')
        check = False
    else:
        print("Loading original dataset")
        COM_NAME = socket.gethostname()
        if COM_NAME == 'Waffle':
            df = pd.read_csv('E:/Git/DMC2016/thirufiles/orders_train.csv',sep=';')
        else:
            df = pd.read_csv('/home/andre/workshop/dmc2016/andrefiles/orders_train.csv',sep=';')
        df = preprocess(df,impute=False,engineerFeatures=check,state=True) #False = dont use imputation.
    return df

def loadTestDataFrame():
    check = True
    if os.path.exists('preprocessed_test.csv'):
        print("Loading feature engineered dataset")
        df = pd.read_csv('preprocessed_test.csv')
        check = False
    else:
        print("Loading original dataset")
        COM_NAME = socket.gethostname()
        if COM_NAME == 'Waffle':
            df = pd.read_csv('E:/Git/DMC2016/thirufiles/orders_class.csv',sep=';')
        else:
            df = pd.read_csv('/home/andre/workshop/dmc2016/andrefiles/orders_class.csv',sep=';')
        df = preprocess(df,impute=False,engineerFeatures=check,state=False) #False = dont use imputation.
    return df

###################################################
#              Preprocessing Methods              #
###################################################
"""
Input:
1) <PD DF> df: pandas dataframe of training data
2) <Boolean> impute: if True, do imputation for missing values rather than dropping
3) <Boolean> engineerFeatures: if True, run feature engineering
4) <Boolean> state: Identifier variable whether train or test set. True = train, False = test
Output:
<PD DF> Processed DF
"""
def preprocess(df,impute,engineerFeatures,state):
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
            df['productGroup'].fillna(-99,inplace=True)
            df['rrp'].fillna(-99,inplace=True)
            df['voucherID'].fillna('MISSING',inplace=True)
            return df
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
    if engineerFeatures:
        print('Running feature engineering..')
        df = featureEngineering(df,state)
    print('Encoding all categorical and object vars to numeric')
    df = oneHotEncode(df)
    df.reset_index(inplace=True,drop=True)
    if state == True: #replace proportion with -99
        print('State is True, running simulateTest..')
        df = simulateTest(df)
    print('Processing done. Saving CSV')
    df.drop('voucherAmount',inplace=True,axis=1) #temporary hack
    if state == 1:
        df.to_csv('preprocessed_train.csv',index=False)
    else:
        df.to_csv('preprocessed_test.csv',index=False)
    return df

###################################################
#           Feature Engineering Methods           #
###################################################
#State: if True, means this is the training data. else, its test data.
def featureEngineering(df,state):
    """
    Returns a column describing how much of the original price was waived by the voucher
    """
    def priceDiscount(df):
        print('Making: priceDiscount')
        priceDiscount = df['voucherAmount'].divide(df['price'],fill_value=0.0)
        priceDiscount[np.isinf(priceDiscount)] = 0.0
        df['priceDiscount'] = priceDiscount
        df['priceDiscount'].fillna(0,inplace=True)
        return df

    """
    Creates new column to indicate if a color is popular or not.
    """
    #Different for train and test --> regenerate for test.
    def colorPopularity(df):
        print('Making: colorPopularity')
        nonlocal state
        if state == True and os.path.exists('pickleFiles/colorMap.pkl'):
            colorMap = joblib.load('pickleFiles/colorMap.pkl')
        elif state==False and os.path.exists('pickleFiles/colorMap_test.pkl'):
            colorMap = joblib.load('pickleFiles/colorMap_test.pkl')
        else:
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
            if state == True:
                joblib.dump(colorMap,'pickleFiles/colorMap.pkl')
            else:
                joblib.dump(colorMap,'pickleFiles/colorMap_test.pkl')
        df['colorPopularity'] = df['colorCode'].map(colorMap)
        return df

    """
    Create totalSpent by customer column as well as averageSpent
    """
    #Different for train and test --> regenerate for test.
    def userSpending(df):
        nonlocal state
        print('Making: userSpending')
        if state == True and os.path.exists('pickleFiles/totalSpent.pkl') and os.path.exists('pickleFiles/averageSpent.pkl') and os.path.exists('pickleFiles/count.pkl'):
            totalSpent = joblib.load('pickleFiles/totalSpent.pkl')
            count = joblib.load('pickleFiles/count.pkl')
            averageSpent = joblib.load('pickleFiles/averageSpent.pkl')
        elif state == False and os.path.exists('pickleFiles/totalSpent_test.pkl'):
            totalSpent = joblib.load('pickleFiles/totalSpent_test.pkl')
            count = joblib.load('pickleFiles/count_test.pkl')
            averageSpent = joblib.load('pickleFiles/averageSpent_test.pkl')
        else:
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
            for i in totalSpent: #slow. theres a better way to do this right
                averageSpent[i] = totalSpent[i] / count[i]

            if state == True:
                joblib.dump(totalSpent,'pickleFiles/totalSpent.pkl')
                joblib.dump(count,'pickleFiles/count.pkl')
                joblib.dump(averageSpent,'pickleFiles/averageSpent.pkl')
            else:
                joblib.dump(totalSpent,'pickleFiles/totalSpent_test.pkl')
                joblib.dump(count,'pickleFiles/count_test.pkl')
                joblib.dump(averageSpent,'pickleFiles/averageSpent_test.pkl')

        df['totalSpent'] = df['customerID'].map(totalSpent)
        df['averageSpent'] = df['customerID'].map(averageSpent)
        df['yearlyExpense'] = df['averageSpent'] / df['totalSpent']
        return df

    # 2 in 1 function to speed up as same loop.
    # 1) Create returnsPerCustomer column, find the total amount of returns per unique
    # customer.
    # 2) create totalPurchases column
    # 3) create purchaseFrequency column
    def purchasesAndReturns(df):
        print('Making: returnsPerCustomer_totalPurchases')
        #if train and both pickle exist, load.
        if state == True and os.path.exists('pickleFiles/returnsPerCustomer.pkl'):
            data  = joblib.load('pickleFiles/returnsPerCustomer.pkl')
            data2 = joblib.load('pickleFiles/totalPurchasesPerCustomer.pkl')
        #if test and both pickle exist, load.
        elif state == False and os.path.exists('pickleFiles/totalPurchasesPerCustomer_test.pkl'):
            data  = joblib.load('pickleFiles/returnsPerCustomer.pkl') #TRAIN DATA
            data2 = joblib.load('pickleFiles/totalPurchasesPerCustomer_test.pkl')
        else: #make new purchases and returnsdata. returns data cannot do with test data.
            if state == False and not os.path.exists('pickleFiles/returnsPerCustomer.pkl'):
                raise Exception('Error with purchasesAndReturns. Cannot make returnsPerCustomer with test data or state set to False')
            elif state == False and os.path.exists('pickleFiles/returnsPerCustomer.pkl'):
                #if test and returns per customer exists.
                data = joblib.load('pickleFiles/returnsPerCustomer.pkl')
                # if train and both pickle dont exist - make it.
            elif state == True and not os.path.exists('pickleFiles/returnsPerCustomer.pkl'):
                data = {}
                for i in df.index:
                    cust = df['customerID'][i]
                    returns = df['returnQuantity'][i]
                    if cust not in data:
                        data[cust] = returns
                    else:
                        data[cust] += returns
                joblib.dump(data,'returnsPerCustomer.pkl')
            data2 = {}
            for i in df.index:
                cust = df['customerID'][i]
                quantity = df['quantity'][i]
                if cust not in data2:
                    data2[cust] = quantity
                else:
                    data2[cust] += quantity
            if state == 1:
                joblib.dump(data2,'pickleFiles/totalPurchasesPerCustomer.pkl')
            else:
                joblib.dump(data2,'pickleFiles/totalPurchasesPerCustomer_test.pkl')

        numMonths = len(df['orderDate'].unique()) #find num months in dataset
        df['returnsPerCustomer'] = df['customerID'].map(data)
        df['totalPurchases'] = df['customerID'].map(data2) #decreases accuracy
        df['purchaseFrequency'] = df['totalPurchases'] / numMonths  #decreases accuracy

        df['returnsPerCustomer'].fillna(-99,inplace=True)
        df['totalPurchases'].fillna(-99,inplace=True)
        df['purchaseFrequency'].fillna(-99,inplace=True)
        return df

    #Creates 2 columns
    # 1) modeSize: most frequent size bought by customer
    # 2) differenceModeSize: difference between modeSize and specific item bought
    #Different for train and test --> regenerate for test.
    def modeSize(df):
        nonlocal state
        print('Making: mostFrequentSize and differenceSize')
        if state == 1 and os.path.exists('pickleFiles/modeSizesBought.pkl'):
            modeSizeData = joblib.load('pickleFiles/modeSizesBought.pkl')
        elif state == 0 and os.path.exists('pickleFiles/modeSizesBought_test.pkl'):
            modeSizeData = joblib.load('pickleFiles/modeSizesBought_test.pkl')
        else:
            allSize = {}
            for i in df.index: #find all sizes purchased by customers
                currCust = df['customerID'][i]
                if currCust not in allSize:
                    allSize[currCust] = [df['sizeCode'][i]]
                else:
                    allSize[currCust].append(df['sizeCode'][i])
            modeSize = {}
            for customer in allSize:
                if customer not in modeSize:
                    mode = Counter(allSize[customer]).most_common(1)[0][0]
                    modeSize[customer] = mode

            if state == 1:
                joblib.dump(modeSize,'pickleFiles/modeSizesBought.pkl')
            else:
                joblib.dump(modeSize,'pickleFiles/modeSizesBought_test.pkl')
            modeSizeData = modeSize

        mostFrequentSize = pd.Series(name= 'mostFrequentSize', index=df.index)
        for i in df.index:
            customer = df['customerID'][i]
            mostFrequentSize.set_value(i,modeSizeData[customer])
        df['modeSize'] = mostFrequentSize
        df['differenceModeSize'] = abs(mostFrequentSize - df['sizeCode'])
        return df

    #Creates 2 columns
    # 1) averageColor: the average colorCode that each customer buys
    # 2) differenceAvgColor: the difference between the specific item bought and averageColor
    #Different for train and test --> regenerate for test.
    def averageColor(df):
        nonlocal state
        print("Making: averageColor")
        if state == 1 and os.path.exists('pickleFiles/averageColor.pkl'):
            averageColor = joblib.load('pickleFiles/averageColor.pkl')
        elif state == 0 and os.path.exists('pickleFiles/averageColor_test.pkl'):
            averageColor = joblib.load('pickleFiles/averageColor_test.pkl')
        else:
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
            if state == 1:
                joblib.dump(averageColor,'pickleFiles/averageColor.pkl')
            else:
                joblib.dump(averageColor,'pickleFiles/averageColor_test.pkl')
        avgcolor = pd.Series(name= 'averageColor', index=df.index)
        for i in df.index:
            customer = df['customerID'][i]
            avgcolor.set_value(i,averageColor[customer])
        df['averageColor'] = avgcolor
        return df

    """
    Creates two columns: popular colors and size for each articleID.
    """
    #Different for train and test --> regenerate for test.
    def articlePopularity(df):
        nonlocal state
        print("Making: articlePopularity")
        if state == 1 and os.path.exists('pickleFiles/popularSizeByArticle.pkl'):
            popColorDic = joblib.load('pickleFiles/popularColorByArticle.pkl')
            popSizeDic = joblib.load('pickleFiles/popularSizeByArticle.pkl')
        elif state == 0 and os.path.exists('pickleFiles/popularSizeByArticle_test.pkl'):
            popColorDic = joblib.load('pickleFiles/popularColorByArticle_test.pkl')
            popSizeDic = joblib.load('pickleFiles/popularSizeByArticle_test.pkl')
        else:
            popSizeDic, popColorDic = {}, {}
            articles = df.groupby('articleID')
            for idx,article in articles:
                if idx not in popSizeDic or idx not in popColorDic:
                    popColorDic[idx] = int(Counter(article['colorCode']).most_common()[0][0])
                    try:
                        popSizeDic[idx] = int(Counter(article['sizeCode']).most_common()[0][0])
                    except:
                        popSizeDic[idx] = np.nan
            if state == 1:
                joblib.dump(popColorDic,'pickleFiles/popularColorByArticle.pkl')
                joblib.dump(popSizeDic,'pickleFiles/popularSizeByArticle.pkl')
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
    #Different for train and test --> regenerate for test.
    def cheapskateItems(df):
        nonlocal state
        print("Making: cheapskateItems")
        if state == 1 and os.path.exists('pickleFiles/voucherToArticle.pkl'):
            voucherDic = joblib.load('pickleFiles/voucherToArticle.pkl')
        elif state == 0 and os.path.exists('pickleFiles/voucherToArticle_test.pkl'):
            voucherDic = joblib.load('pickleFiles/voucherToArticle_test.pkl')
        else:
            voucherDic = {}
            vouchers = df.groupby('voucherID')
            for idx,voucher in vouchers:
                if idx not in voucherDic:
                    voucherDic[idx] = Counter(voucher['articleID']).most_common()[0][0]
            if state == 1:
                joblib.dump(voucherDic,'pickleFiles/voucherToArticle.pkl')
            else:
                joblib.dump(voucherDic,'pickleFiles/voucherToArticle_test.pkl')

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
        nonlocal state
        print("Making: varianceInProductGroups")
        if state == 1 and os.path.exists('pickleFiles/colorStd.pkl') and os.path.exists('pickleFiles/sizeStd.pkl'):
            sizeStd = joblib.load('pickleFiles/sizeStd.pkl')
            colorStd = joblib.load('pickleFiles/colorStd.pkl')
        elif state == 0 and os.path.exists('pickleFiles/colorStd_test.pkl') and os.path.exists('pickleFiles/sizeStd_test.pkl'):
            sizeStd = joblib.load('pickleFiles/sizeStd_test.pkl')
            colorStd = joblib.load('pickleFiles/colorStd_test.pkl')
        else:
            products = df.groupby('productGroup')
            sizeStd, colorStd = {},{}
            for idx,product in products:
                if idx not in sizeStd or idx not in colorStd:
                    size = np.std(list(Counter(product['sizeCode']).values()))
                    color = np.std(list(Counter(product['colorCode']).values()))
                    sizeStd[idx] = size
                    colorStd[idx] = color
            if state == 1:
                joblib.dump(sizeStd,'pickleFiles/sizeStd.pkl')
                joblib.dump(colorStd,'pickleFiles/colorStd.pkl')
            else:
                joblib.dump(sizeStd,'pickleFiles/sizeStd_test.pkl')
                joblib.dump(colorStd,'pickleFiles/colorStd_test.pkl')
        df['sizeStd'] = df['productGroup'].map(sizeStd)
        df['colorStd'] = df['productGroup'].map(colorStd)
        return df
    #Different for train and test --> regenerate for test.
    def isRepeatCustomer(df):
        nonlocal state
        print("Making: isRepeatCustomer")
        if state == 1 and os.path.exists('pickleFiles/repeatCustomer.pkl'):
            d = joblib.load('pickleFiles/repeatCustomer.pkl')
        elif state == 0 and os.path.exists('pickleFiles/repeatCustomer_test.pkl'):
            d = joblib.load('pickleFiles/repeatCustomer_test.pkl')
        else:
            d = {}
            for i in df.index:
                idx = df['customerID'][i]
                if idx not in d:
                    d[idx] = 1
                else:
                    d[idx] += 1
            if state == 1:
                joblib.dump(d,'pickleFiles/repeatCustomer.pkl')
            else:
                joblib.dump(d,'pickleFiles/repeatCustomer_test.pkl')
        singlePurchase = {k:v-1 for k,v in d.items()} # hackish trick to force keys with value of 1 to 0, so that it evaluates to false
        df['repeatCustomer'] = [1 if singlePurchase[df['customerID'][j]] else 0 for j in df.index]
        return df
    #Different for train and test --> regenerate for test.
    def weekendWeekday(df):
        nonlocal state
        print("Making: weekendWeekday")
        if state == 1 and os.path.exists('pickleFiles/dayOfTheWeek.pkl'):
            dayOfTheWeek = joblib.load('pickleFiles/dayOfTheWeek.pkl')
        elif state == 0 and os.path.exists('pickleFiles/dayOfTheWeek_test.pkl'):
            dayOfTheWeek = joblib.load('pickleFiles/dayOfTheWeek_test.pkl')
        else:
            dateObject = pd.DatetimeIndex(pd.to_datetime(df['orderDate']))
            dayOfTheWeek = {}
            for i in df.index:
                currDate = df['orderDate'][i]
                if currDate not in dayOfTheWeek:
                    dayInteger = dateObject[i].weekday()
                    if dayInteger == 5 or dayInteger == 6: # weekends
                        dayOfTheWeek[currDate] = 1
                    else: # weekdays
                        dayOfTheWeek[currDate] = 0
            if state == 1:
                joblib.dump(dayOfTheWeek,'pickleFiles/dayOfTheWeek.pkl')
            else:
                joblib.dump(dayOfTheWeek,'pickleFiles/dayOfTheWeek_test.pkl')
        df['isWeekend'] = df['orderDate'].map(dayOfTheWeek)
        return df

    ## 1. TEST DONT HAVE. USE TRAIN.
    # 2. Runs fast enough, dont need pickle.
    def highReturnItem(df):
        if not os.path.exists('pickleFiles/returnRates.pkl'):
            articles = df.groupby('articleID')
            returnRates = {}
            for idx,article in articles:
                returnRates[idx] = sum(article['returnQuantity'])

            joblib.dump(returnRates,'pickleFiles/returnRates.pkl')
        else:
            returnRates = joblib.load('pickleFiles/returnRates.pkl')

        df['returnRates'] = df['articleID'].map(returnRates)
        mean = df['returnRates'].mean()
        df.ix[df.returnRates > mean,'returnRates'] = 1
        df.ix[df.returnRates <= mean,'returnRates'] = 0
        df['returnRates'].fillna(-99,inplace=True)
        return df

    ## TEST DONT HAVE. USE TRAIN.
    def customerReturnSpecificItem(df):
        if not os.path.exists('pickleFiles/likelyreturn.pkl'):
            dic = {}
            for i in df.index:
                currCustomer = df['customerID'][i]
                if currCustomer not in dic:
                    dic[currCustomer] = {df['articleID'][i]:df['returnQuantity'][i]}
                else:
                    article = df['articleID'][i]
                    if article not in dic[currCustomer]:
                        dic[currCustomer][article] = df['returnQuantity'][i]
                    else:
                        dic[currCustomer][article] += df['returnQuantity'][i]
        else:
            dic = joblib.load('pickleFiles/likelyreturn.pkl')
        likelyReturn = pd.Series(name= 'likelyReturn', index=df.index)
        for i in df.index:
            currItem = df['articleID'][i]
            currCustomer = df['customerID'][i]
            try:
                if dic[currCustomer][currItem] > 0:
                    likelyReturn.set_value(i,1)
                else:
                    likelyReturn.set_value(i,0)
            except:
                likelyReturn.set_value(i,-99)
        df['customerSpecificReturn'] = likelyReturn
        return df

    ## TEST DONT HAVE. USE TRAIN
    # makes 3 columns, similar function. eg, for size, it finds all the products every customer
    # has returned, and which one he kept. Then makes a col called likelyReturnSize, where its
    # 1 if return before, 0 if keep, -99 if the item size he bought is new (unknown)
    def returnSizePdtgrpColor(train):
        if not os.path.exists('pickleFiles/likelyreturnSize.pkl'):
            sizeDic = {}
            colorDic={}
            pdtGroup = {}
            for j in train.index:
                currCust = train['customerID'][j]
                if currCust in sizeDic:
                    continue
                newdf = train[train['customerID'] == currCust] #find all one shot
                returnYes = newdf[newdf['returnQuantity'] > 0]
                returnNo = newdf[newdf['returnQuantity'] == 0]

                returnedSize = Counter(returnYes['sizeCode']).most_common()
                returnedColor = Counter(returnYes['colorCode']).most_common()
                returnedPdtGrp = Counter(returnYes['productGroup']).most_common()
                keptSize = Counter(returnNo['sizeCode']).most_common()
                keptColor = Counter(returnNo['colorCode']).most_common()
                keptPdtGrp = Counter(returnNo['productGroup']).most_common()

                keepList = [keptSize,keptColor,keptPdtGrp]
                returnList = [returnedSize,returnedColor,returnedPdtGrp]

                for i in range(len(keepList)):
                    keepList[i] = dict((x,y) for x,y in keepList[i])
                    returnList[i] = dict((x,y) for x,y in returnList[i])

                    for key in returnList[i]: #for every key like returnedSize,
                        if key in keepList[i]: #return - keep. if +ve, means return MORE
                            result =  returnList[i][key] - keepList[i][key]
                        else: #if key not in returnList,
                            result = returnList[i][key]
                        #At this point, the only remaining items are those only in keepList
                        if result > 0:
                            decision = 1 #return
                        elif result <0:
                            decision = 0 #keep
                        else: #if draw
                            decision = -99

                        if i ==0:
                            sizeDic[currCust] = {key:decision}
                        elif i==1:
                            colorDic[currCust] = {key:decision}
                        else:
                            pdtGroup[currCust] = {key:decision}

                    for key,result in keepList[i].items():
                        if key not in returnList[i]:
                            if i ==0:
                                sizeDic[currCust] = {key:0}
                            elif i==1:
                                colorDic[currCust] = {key:0}
                            else:
                                pdtGroup[currCust] = {key:0}
            joblib.dump(sizeDic,'pickleFiles/likelyreturnSize.pkl')
            joblib.dump(colorDic,'pickleFiles/likelyreturnColor.pkl')
            joblib.dump(pdtGroup,'pickleFiles/likelyreturnPdtGrp.pkl')
        else:
            sizeDic = joblib.load('pickleFiles/likelyreturnSize.pkl')
            colorDic=joblib.load('pickleFiles/likelyreturnColor.pkl')
            pdtGroup = joblib.load('pickleFiles/likelyreturnPdtGrp.pkl')

        #find difference between kept and not kept. make final column
        likelyReturnSize = pd.Series(name= 'likelysize', index=train.index)
        likelyReturnPdtGrp = pd.Series(name= 'likelypdtgrp', index=train.index)
        likelyReturnColor = pd.Series(name= 'likelyreturncolor', index=train.index)
        for i in train.index:
            currCust = train['customerID'][i]
            if currCust not in sizeDic:
                likelyReturnSize.set_value(i,-99)
                likelyReturnPdtGrp.set_value(i,-99)
                likelyReturnColor.set_value(i,-99)
            else:
                size = train['sizeCode'][i]
                grp = train['productGroup'][i]
                color = train['colorCode'][i]
                if size in sizeDic[currCust]:
                    likelyReturnSize.set_value(i,sizeDic[currCust][size])
                else:
                    sizeSet = set(sizeDic[currCust].keys())
                    likelyReturnSize.set_value(i,min(sizeSet,key=lambda x:abs(x-size)))
                if color in colorDic[currCust]:
                    likelyReturnColor.set_value(i,colorDic[currCust][color])
                else:
                    likelyReturnColor.set_value(i,-99)
                if grp in pdtGroup[currCust]:
                    likelyReturnPdtGrp.set_value(i,pdtGroup[currCust][grp])
                else:
                    likelyReturnPdtGrp.set_value(i,-99)

        train['likelyReturnSize'] = likelyReturnSize
        train['likelyReturnPdtGrp'] = likelyReturnPdtGrp
        train['likelyReturnColor'] = likelyReturnColor
        return train

    df = purchasesAndReturns(df)
    df = userSpending(df)
    df = priceDiscount(df)
    df = colorPopularity(df)
    df = modeSize(df) #ONLY modeSize. Excluded diffModeSize, diffAvgSize, avgSize
    df = averageColor(df) #ONLY averageColor. Excluded diffAvgColor
    df = articlePopularity(df)
    df = cheapskateItems(df)
    df = varianceInProductGroups(df)
    df = isRepeatCustomer(df) #boolean version
    df = weekendWeekday(df)
    df = highReturnItem(df)
    df = customerReturnSpecificItem(df)
    df = returnSizePdtgrpColor(df)
    print('Feature Engineering Done')
    return df

def simulateTest(df):
     import random as rand
     length = len(df)
     #customer specific -> 42% unknown customers
     randomList = rand.sample(range(1,length), int(length*(0.42)))
     df['returnsPerCustomer'][randomList] = -99
     df['likelyReturnSize'][randomList] = -99
     df['likelyReturnPdtGrp'][randomList] = -99
     df['likelyReturnColor'][randomList] = -99
     #for customerSpecific return --> only 2% bought same products in test
     randomList = rand.sample(range(1,length), int(length*(0.98)))
     df['customerSpecificReturn'][randomList] = -99
     #item specific: only 72% known items
     randomList = rand.sample(range(1,length), int(length*(0.28)))
     df['returnRates'][randomList] = -99
     return df


"""
Input:
1) <PD DF> df: pandas dataframe

Output:
1) <numpy array> dataset: the features of the training set
2) <numpy array> target: the labels of the training set
"""
def splitDatasetTarget(df):
    dataset = df.drop('returnQuantity', axis=1)
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

#takes in dataset,target after stratifiedSampleGenerator
#
def testFeatureAccuracy2(dataset,target):
    #dropped orderID and quantity as per preprocessing.
    originalCols = ['orderDate', 'articleID', 'colorCode', 'sizeCode', 'productGroup',
                    'price', 'rrp', 'voucherID','customerID', 'deviceID',
                    'paymentMethod']
    newFeatures = set(list(dataset.columns))
    newFeatures = list(newFeatures.difference(originalCols))
    accuracy = None
    numNewFeatures = len(newFeatures) #find num new features
    classifier = xgBoost()
    keep = []
    discard = []
    #find base score
    print('Finding base Score')
    trainx,testx,trainy,testy = train_test_split(dataset,target,test_size=0.2)
    classifier.fit(trainx[originalCols],trainy, early_stopping_rounds=25, eval_metric="merror", eval_set=[(testx[originalCols], testy)])
    accuracy = classifier.score(testx[originalCols],testy)
    #loop through new features, add one by one. if improve, keep. else discard
    for i in range(numNewFeatures):
        print('Doing feature '+str(i) +' out of total '+str(numNewFeatures))
        originalCols.append(newFeatures[i])
        classifier.fit(trainx[originalCols],trainy, early_stopping_rounds=25,
                       eval_metric="merror", eval_set=[(testx[originalCols], testy)])

        score = classifier.score(testx[originalCols],testy)
        if accuracy < score: #if improve accuracy
            accuracy = score
            keep.append(newFeatures[i])
        else:
            originalCols.remove(newFeatures[i])
            discard.append(newFeatures[i])
    print('Final accuracy is '+str(accuracy))
    ##In case
    joblib.dump(originalCols,'colstokeep.pkl')
    return originalCols,keep,discard


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
    global accuracy
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
    # cols needed: orderID,articleID,colorCode,sizeCode,prediction
    pass

# tuning TODO list
# 1. tune max depth: original (8), best (8), fail (7,9)
# 2. tune min child weight
# 2. tune gamma
# 3. tune subsample
# 4. tune colsample_bytree
# 4. tune regularization alpha param
# 5. reduce learning rate
def tuneParameters(dataset,target):
    #find base accuracy first
    trainx,testx,trainy,testy = train_test_split(dataset,target,test_size=0.2)
    classifier = xgb.XGBClassifier(max_depth = 9,n_estimators=300,nthread=8,seed=1,silent=1,
                            objective= 'multi:softmax',learning_rate=0.1,subsample=0.9)
    classifier.fit(trainx,trainy, early_stopping_rounds=25,
                       eval_metric="merror", eval_set=[(testx, testy)])
    testAccuracy = classifier.score(testx,testy)
    params = classifier.get_params()
    fileName = 'tuneParameters.csv'
    if os.path.isfile(fileName) == False:
        with open(fileName,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Params','Accuracy','Tuning'])
    with open(fileName,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([params,testAccuracy,"Max depth"])

def run():
    train = loadDataFrame()
    global datasetSize
    datasetSize = len(train)
    dataset,target = splitDatasetTarget(train)
    dataset,target = stratifiedSampleGenerator(dataset,target,test_size=0.25)
    # testFeatureAccuracy(dataset,target)
    # finalCols,keepList,discardList = testFeatureAccuracy2(dataset,target)
    finalCols = joblib.load('colstokeep.pkl')
    finalCols.extend(['likelyReturnSize','likelyReturnColor','likelyReturnPdtGrp'])
    for i in dataset.columns:
        dataset[i].fillna(-99,inplace=True)
    # tuneParameters(dataset[finalCols],target)
    # clfs = [xgBoost(),randomForest(),extraTrees(),kNN(),neuralNetwork()]
    clfs = [xgBoost(),randomForest(),extraTrees()]
    clfs = accuracyChecker(dataset[finalCols],target,clfs,cross_val=False,ensemble = True,record = True,predictTest=False) # Dont use CV, Yes ensemble, Yes Record.

    #test = loadTestDataFrame()
if __name__ == '__main__':
	run()
