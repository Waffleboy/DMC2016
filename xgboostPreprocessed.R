library(xgboost)
library(Matrix)
library(caret)
setwd("/home/andre/workshop/dmc2016")
# setwd("path/to/thirufiles/here")
train <- read.csv("preprocessed_train.csv")

# train = train[c('orderDate','articleID','colorCode','sizeCode','productGroup','customerID','deviceID','paymentMethod','voucherID','modeSize','averageColor','priceDiscount','quantity','cheapArticle','price','averageSpent','differenceModeSize','purchaseFrequency','repeatCustomer', 'totalSpent','rrp','totalPurchases','returnsPerCustomer','customerSpecificReturn','yearlyExpense','returnQuantity')] # thirucols

datasetSize = nrow(train) #for later computation

##SAMPLE of 0.2 of entire dataset
SAMPLE_SIZE = 0.2
train <- train[sample(1:nrow(train), nrow(train)*SAMPLE_SIZE, replace=FALSE),]

train <- train[c('orderDate', 'articleID', 'colorCode', 'sizeCode', 'productGroup', 'price', 'rrp', 'voucherID', 'customerID', 'deviceID', 'paymentMethod', 'repeatCustomer', 'purchaseFrequency', 'averageSpent', 'differenceModeSize', 'yearlyExpense', 'colorPopularity', 'customerSpecificReturn', 'returnsPerCustomer', 'totalPurchases','likelyReturnSize','likelyReturnColor','likelyReturnPdtGrp','returnQuantity')]

#split to dataset and target label
labels = train['returnQuantity'] 
train = train[-grep('returnQuantity', colnames(train))]

#CROSS VALIDATION CHECK
# dtrain <- xgb.DMatrix(data=train, label=labels)
# bst <- xgb.cv(data = dtrain, label = labels, nfold = 5,
#          nrounds = 500, objective = "multi:softmax",num_class = 6,
#          early.stop.round = 25, maximize = FALSE,nthread=8)

##SPLIT TO TRAIN  TEST
PROPORTION = 0.75 #set to 0.75 train

smp_size <- floor(PROPORTION * nrow(train))
## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)
trainx <- train[train_ind, ]
testx <- train[-train_ind, ]

trainy <- labels[train_ind, ]
testy <- labels[-train_ind, ]

testSize = nrow(testx)

train <- sparse.model.matrix(trainy ~ ., data = trainx)
test <- sparse.model.matrix(testy ~ ., data = testx)

dtrain <- xgb.DMatrix(data=train, label=trainy)
dtest <- xgb.DMatrix(data=test, label=testy)

watchlist <- list(val=dtest,train=dtrain)

param <- list(  objective   = "multi:softmax", 
                num_class   = 6,
                eval_metric = "merror",
                eta         = 0.1,
                max_depth   = 8,
                subsample   = 0.9,
                nthread     = 8,
                set.seed    = 123,
                min_child_weight = 7,
                gamma       = 0.25,
                subsample   = 1.0,
                colsample_bytree = 0.8)

# clf <- xgb.train(   params              = param, 
#                     data                = dtrain, 
#                     nrounds             = 700, 
#                     verbose             = 1,
#                     watchlist           = watchlist,
#                     maximize            = FALSE,
#                     early.stop.round    = 25)

tuneGrid <- expand.grid(subsample = c(0.8,0.9,1.0), colsample_bytree = c(0.8,0.9,1.0))

gridSearch <- apply(tuneGrid, 1, function(parameterList){

    param$subsample = parameterList[["subsample"]]
    param$colsample_bytree = parameterList[["colsample_bytree"]]

    clf <- xgb.train(   params              = param, 
                        data                = dtrain, 
                        nrounds             = 700, 
                        verbose             = 1,
                        watchlist           = watchlist,
                        maximize            = FALSE,
                        early.stop.round    = 25)

    
    label = getinfo(dtest, "label")
    pred <- predict(clf, dtest)
    err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
    return(c(err,param$subsample,param$colsample_bytree))

})

print(gridSearch)

#### Competition metric functions ###
computeError = function(predicted,target){
  return (sum(abs(predicted-target)))
}

computeErrorScaled = function(computeError,datasetSize,testSize){
  return ( (computeError*datasetSize) / testSize )
}
######################################################

#Predict Data and score it
# label = getinfo(dtest, "label")
# pred <- predict(clf, dtest)
# err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
# print(paste("test-error=", err))
# print(paste("Accuracy =", 1- err))

#Compute competiton metric
# error = computeError(pred,testy)
# scaledError = computeErrorScaled(error,datasetSize,testSize)

# print(paste('Competition Error is', as.character(error)))
# print(paste('Scaled Competition Error is', as.character(scaledError)))