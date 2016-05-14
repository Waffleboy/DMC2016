library(xgboost)
library(Matrix)

#Thirucols only, 2.2m rows, no additional features like returnsize, accuracyL 0.71807

## Insert your dataset here.
train <- read.csv("E:/git/DMC2016/preprocessed_train.csv")


train = train[c('orderDate','articleID','colorCode','sizeCode','productGroup','customerID','deviceID','paymentMethod','voucherID',
         'modeSize','averageColor','priceDiscount','quantity','cheapArticle','price','averageSpent',
         'differenceModeSize','purchaseFrequency','repeatCustomer','totalSpent','rrp','totalPurchases','returnsPerCustomer',
         'customerSpecificReturn','yearlyExpense','returnQuantity')]

datasetSize = nrow(train) #for later computation

##SAMPLE of 0.2 of entire dataset
#SAMPLE_SIZE = 0.2
#train <- train[sample(1:nrow(train), nrow(train)*SAMPLE_SIZE,
#                       replace=FALSE),]

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

watchlist <- list(test=dtest,train=dtrain)

param <- list(  objective           = "multi:softmax", 
                num_class           = 6,
                eval_metric         = "merror",
                eta                 = 0.1,
                max_depth           = 8,
                subsample           = 0.9,
                nthread             = 8,
                set.seed            = 123
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 700, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    early.stop.round    = 25
)

#### Competition metric functions ###
computeError = function(predicted,target){
  return (sum(abs(predicted-target)))
}

computeErrorScaled = function(computeError,datasetSize,testSize){
  return ( (computeError*datasetSize) / testSize )
}
######################################################

#Predict Data and score it
label = getinfo(dtest, "label")
pred <- predict(clf, dtest)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))
print(paste("Accuracy =", 1- err))

#Compute competiton metric
error = computeError(pred,testy)
scaledError = computeErrorScaled(error,datasetSize,testSize)

print(paste('Competition Error is', as.character(error)))
print(paste('Scaled Competition Error is', as.character(scaledError)))
