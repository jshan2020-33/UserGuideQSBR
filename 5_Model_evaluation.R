## Description: 
# predT, predV respects the predicted k of the training and testing set, respectively.
# obsT, obsV respects the observed k of the training and testing set, respectively.
# predT, predV, obsT, and obsV be used to calculate multiple criteria for evaluating model.


## Calculating R2, Q2test, RMSEtrain and RMSEtest.
library("caret")
postResample(predT,obsT)
postResample(predV,obsV)


## Calculating NSE and PBIAS
library("hydroGOF")
NSEtrain = NSE(predT,obsT)
NSEtest = NSE(predV,obsV)
PBIAStrain = pbias(predT,obsT)
PBIAStest = pbias(predV,obsV)


## Calculating CCC
library("DescTools")
train.ccc <- CCC(obsT,predT, ci = "z-transform",
   conf.level = 0.95)
	 CCCtrain = train.ccc$rho.c[,1]
	 CCCtraindown = train.ccc$rho.c[,2]
	 CCCtrainup = train.ccc$rho.c[,3]
test.ccc <- CCC(obsV,predV, ci = "z-transform",
   conf.level = 0.95)
	 CCCtest = test.ccc$rho.c[,1]
	 CCCtestdown = test.ccc$rho.c[,2]
	 CCCtestup = test.ccc$rho.c[,3]


## Calculating bootstrapping coefficient (Q2BOOT) and bootstrapping root mean square error (RMSEBOOT)
library(caret)

# To primary MLR model
train.control <- trainControl(method = "boot", number = 100)
model <- train(Log10P~.,data=data_end, method = "lm",
               trControl = train.control)

# To primary XGBoost model
params <- expand.grid(nrounds=500,
	max_depth = 1,
	eta = 0.1,
	gamma = 0.01,
	min_child_weight = 1,
	subsample = 0.75,
	colsample_bytree = 1)
train.control <- trainControl(method = "boot", number = 100)
model <- train(PrimaryObs~.,
			   data = Train_data,
			   method = "xgbTree",
               trControl = train.control,
			   tuneGrid = params)

# To primary SVM model
params <- expand.grid(C=13.3267149925232)
train.control <- trainControl(method = "boot", number = 100)
model <- train(PrimaryObs~.,
			   data = train_data,
			   method = "svmLinear",
               trControl = train.control,
			   tuneGrid = params)

print(model)