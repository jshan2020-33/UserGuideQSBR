library(caret)
library(caTools)
library(MASS)
library(Matrix)
library(xgboost)

############################## Import dataset
setwd("D:/UserGuideQSBR")
load("/RawData/DataSet.RData")

home = "D:/UserGuideQSBR/"

type =	'2nearZeroVar_PU_005_NoNA_traintest'

#### Set different options depending on the predicted object
target = 'Bio_FromAll_PU_traintest'
Target = 'Primary and Ultimate Biodegradability'
lim_min = 1
lim_max = 5
lim = 0.3
at = c(1,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0)
nDescnum = 13 
CanCha = 4
ganggan = 2
PO = 0.3

	
#### Read descriptor information
Perform = read.csv(file=paste(home,"StepswiseMLR\\ML_Steps_",type, ".csv", sep=""),head = TRUE)

AddDesc = Perform[Perform$AddorRem=='+',]$Parameter
RemDesc = Perform[Perform$AddorRem=='x',]$Parameter

## remove RemDesc in AddDesc
Desc = AddDesc # In order not to destroy AddDesc
for(i in RemDesc){Desc = Desc[-which(Desc==i)]} # Loop to delete variables in RemDesc one by one

#### Obtain the hyperparameters of Unifed model based on XGBoost
MLSteps = read.csv(file =paste(home,"Unifed_XGBoost_hyperparameters\\XGB_MLSteps_DescChange_10KFold_",
								target,".csv",sep = ""))[nDescnum-1,]
nrounds = MLSteps[,'nround']
nthreads = 3
lambda = MLSteps[,'lambda']
max_depth = MLSteps[,'max_depth']
eta = MLSteps[,'eta']
gamma = MLSteps[,'gamma']
min_child_weight = MLSteps[,'min_child_weight']
subsample = MLSteps[,'subsample']
colsample_bytree = MLSteps[,'colsample_bytree']
colsample_bylevel = MLSteps[,'colsample_bylevel']
	
#### Generate data set
dataP = data_ALL[,c("PrimaryObs",Desc[2:nDescnum])]
dataU = data_ALL[,c("UltimateObs",Desc[2:nDescnum])]

dataP = cbind(dataP,PUlabels = rep(1,173))
dataU = cbind(dataU,PUlabels = rep(0,173))

names(dataP)[1] = "Bio"
names(dataU)[1] = "Bio"
Data = rbind(dataP,dataU)

set.seed(22)
library(caTools)
TM=sample.split(Data$Bio,4/5) 
Train_data = Data[TM,]
Vaild_data = Data[!TM,]


## Data preprocessing

# Transform the independent variable into a sparse matrix
train_matrix <- sparse.model.matrix(Bio ~ .-1, data = Train_data)
test_matrix <- sparse.model.matrix(Bio ~ .-1, data = Vaild_data)

# Merge the independent and dependent variables into a list
train_fin <- list(data=train_matrix,label=Train_data[,1]) 
test_fin <- list(data=test_matrix,label=Vaild_data[,1]) 

dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label) 
dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)


#### Modeling and forecasting
set.seed(22)
model <- xgboost(data = dtrain, nround=nrounds,nthread = nthreads,
	lambda = lambda,
	max_depth = max_depth,
	eta = eta,
	gamma = gamma,
	min_child_weight = min_child_weight,
	subsample = subsample,
	colsample_bytree = colsample_bytree,
	colsample_bylevel = colsample_bylevel,
	verbose = 0)

# Geting predicted k for training and testing set
predT=predict(model, newdata = dtrain)
predV=predict(model, newdata = dtest)
obsT=Train_data$Bio
obsV=Vaild_data$Bio
