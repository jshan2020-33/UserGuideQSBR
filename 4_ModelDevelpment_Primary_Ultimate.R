############################## Import dataset
setwd("D:/UserGuideQSBR")
load("/RawData/DataSet.RData")

home = "D:/UserGuideQSBR/"


#### 1. MLR ###############################################################################################
library(caTools)
set.seed(22) # Dividing the data set randomly
TM=sample.split(data_ALL$Log10P,4/5) 
train_data=data_ALL[TM,]
test_data=data_ALL[!TM,]

##### 1.1 MLR_P ############################################################
library(tidyverse)
data_end=select(train_data,c(Log10P,
							GraphFP245,
							PatternFP695_RDKit,
							C1SP3,
							AATS0v,
							MDEO.12,
							MHFP186_RDKit,
							ExtFP594,
							ecfp8_1379_open))
LM = lm(Log10P~.,data=data_end)

# Geting predicted k for training and testing set
obsT=10^train_data$Log10P
predT=10^predict(LM,train_data)
obsV=10^test_data$Log10P
predV=10^predict(LM,test_data)


#### 1.2 MLR_U #############################################################
data_end=select(train_data,c(Log10U,
							No..of.Rotatable.Bonds,
							N.Ratio,
							Halogen.Ratio,
							TDE,
							A,
							logSw,
							C1SP3,
							AATS3p))
LM = lm(Log10U~.,data=data_end)

# Geting predicted k for training and testing set
obsT=10^train_data$Log10U
predT=10^predict(LM,train_data)
obsV=10^test_data$Log10U
predV=10^predict(LM,test_data)




##### 2. XGBoost ############################################################################
############################################## 2.1 XGBoost_P_nDesc=9
# XGBoost_Primary
algorithm = 'XGBoost'

Perform = read.csv(file= paste(home,'StepswiseMLR/ML_Steps_2nearZeroVar_P_005_NoNA.csv',sep = ""),head = TRUE)

AddDesc = Perform[Perform$AddorRem=='+',]$Parameter
RemDesc = Perform[Perform$AddorRem=='x',]$Parameter

# # Delete RemDesc in AddDesc
Desc = AddDesc
for(i in RemDesc){Desc = Desc[-which(Desc==i)]}

i = 9
lim_min = 2
lim_max = 5
lim = 0.3
at = c(2.0,2.5,3.0,3.5,4.0,4.5,5.0)
target = 'PrimaryObs'

library(caTools)
Data = data_ALL[,c(target,Desc[1:i])]
set.seed(22)
TM=sample.split(Data[,1],4/5)
Train_data = Data[TM,]
Vaild_data = Data[!TM,]

# Data preprocessing
library(xgboost)
library(Matrix)
train_matrix <- sparse.model.matrix(PrimaryObs ~ .-1, data = Train_data)
test_matrix <- sparse.model.matrix(PrimaryObs ~ .-1, data = Vaild_data)
train_fin <- list(data=train_matrix,label=Train_data[,1])
test_fin <- list(data=test_matrix,label=Vaild_data[,1])
dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label)
dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)

set.seed(22)
model <- xgboost(data = dtrain, nrounds = 500,nthread = 3,
	lambda = 0,
	max_depth = 1,
	eta = 0.1,
	gamma = 0.01,
	min_child_weight = 1,
	subsample = 0.75,
	colsample_bytree = 1,
	colsample_bylevel = 0.4,
	verbose=0)

# Geting predicted k for training and testing set
predT=predict(model, newdata = dtrain)
predV=predict(model, newdata = dtest)
obsT=Train_data$PrimaryObs
obsV=Vaild_data$PrimaryObs



####################################################### 2.2 XGBoost_U_nDesc=15
Perform = read.csv(file= paste(home,'StepswiseMLR/ML_Steps_2nearZeroVar_U_005_NoNA.csv',sep=""),head = TRUE)

AddDesc = Perform[Perform$AddorRem=='+',]$Parameter
RemDesc = Perform[Perform$AddorRem=='x',]$Parameter

# Delete RemDesc in AddDesc
Desc = AddDesc 
for(i in RemDesc){Desc = Desc[-which(Desc==i)]}

i = 15
lim_min = 1
lim_max = 4.5
lim = 0.35
at = c(1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5)
target = 'UltimateObs'

library(caTools)
Data = data_ALL[,c(target,Desc[1:i])]
set.seed(22)
TM=sample.split(Data[,1],4/5)
Train_data = Data[TM,]
Vaild_data = Data[!TM,]

# Data preprocessing
library(xgboost)
library(Matrix)
train_matrix <- sparse.model.matrix(UltimateObs ~ .-1, data = Train_data)
test_matrix <- sparse.model.matrix(UltimateObs ~ .-1, data = Vaild_data)
train_fin <- list(data=train_matrix,label=Train_data[,1])
test_fin <- list(data=test_matrix,label=Vaild_data[,1])
dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label)
dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)

# Developing ultimate XGBoost model
set.seed(22)
model <- xgboost(data = dtrain, nround=500,nthread = 3,
	lambda = 0,
	max_depth = 1,
	eta = 0.1,
	gamma = 0,
	min_child_weight = 1,
	subsample = 0.75,
	colsample_bytree = 1,
	colsample_bylevel = 0.4,
	verbose=0)

# Geting predicted k for training and testing set
predT=predict(model, newdata = dtrain)
predV=predict(model, newdata = dtest)
obsT=Train_data$UltimateObs
obsV=Vaild_data$UltimateObs




##### 3. SVM #######################################################################################################################

#######################################################3.1 SVM_P_nDesc=38
#### SVM_Primary
algorithm = "SVM"
Perform = read.csv(file= paste(home,'StepswiseMLR/ML_Steps_2nearZeroVar_P_005_NoNA.csv',sep=""),head = TRUE)

AddDesc = Perform[Perform$AddorRem=='+',]$Parameter
RemDesc = Perform[Perform$AddorRem=='x',]$Parameter

# Delete RemDesc in AddDesc
Desc = AddDesc
for(i in RemDesc){Desc = Desc[-which(Desc==i)]}

i = 38
lim_min = 2
lim_max = 5
lim = 0.3
at = c(2.0,2.5,3.0,3.5,4.0,4.5,5.0)
target = 'PrimaryObs'

library(caTools)
Data = data_ALL[,c(target,Desc[1:i])]
set.seed(22)
TM=sample.split(Data[,1],4/5) 
PU_Train_noweeks = Data[TM,]
PU_Vaild_noweeks = Data[!TM,]

# Data preprocessing of 01
center = sweep(PU_Train_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
R = apply(PU_Train_noweeks[,-1],2,max) - apply(PU_Train_noweeks[,-1],2,min)
data_x_01_T = sweep(center,2,R,"/")
train_data = cbind(PrimaryObs=PU_Train_noweeks$PrimaryObs,data_x_01_T)

center_V = sweep(PU_Vaild_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
data_x_01_V = sweep(center_V,2,R,"/")
Vaild_data = cbind(PrimaryObs=PU_Vaild_noweeks$PrimaryObs,data_x_01_V)

# Developing primary SVMt model
model=svm( PrimaryObs~., data = train_data, type = "eps-regression", cost=13.3267149925232,kernel = "linear",scale=0)

# Geting predicted k for training and testing set
predT = predict(model, newdata = train_data)
obsT = PU_Train_noweeks$PrimaryObs
predV = predict(model, newdata = Vaild_data)
obsV = PU_Vaild_noweeks$PrimaryObs


#####################################################3.2 SVM_U_nDesc=44
# SVM_Ultimate
algorithm = "SVM"
Perform = read.csv(file= paste(home,'StepswiseMLR/ML_Steps_2nearZeroVar_U_005_NoNA.csv',sep=""),head = TRUE)

AddDesc = Perform[Perform$AddorRem=='+',]$Parameter
RemDesc = Perform[Perform$AddorRem=='x',]$Parameter

# Delete RemDesc in AddDesc
Desc = AddDesc 
for(i in RemDesc){Desc = Desc[-which(Desc==i)]}

i = 44
lim_min = 1
lim_max = 4.5
lim = 0.35
at = c(1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5)
target = 'UltimateObs'

library(caTools)
Data = data_ALL[,c(target,Desc[1:i])]
set.seed(22)
TM=sample.split(Data[,1],4/5)
PU_Train_noweeks = Data[TM,]
PU_Vaild_noweeks = Data[!TM,]


# Data preprocessing of 01
center = sweep(PU_Train_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
R = apply(PU_Train_noweeks[,-1],2,max) - apply(PU_Train_noweeks[,-1],2,min)
data_x_01_T = sweep(center,2,R,"/")
train_data = cbind(UltimateObs=PU_Train_noweeks$UltimateObs,data_x_01_T)

center_V = sweep(PU_Vaild_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
data_x_01_V = sweep(center_V,2,R,"/")
Vaild_data = cbind(UltimateObs=PU_Vaild_noweeks$UltimateObs,data_x_01_V)

# Developing ultimate SVMt model
model=svm( UltimateObs~., data = train_data, type = "eps-regression", cost=21.593137845397,kernel = "linear",scale=0)

# Geting predicted k for training and testing setpredT = predict(model, newdata = train_data)
predT = predict(model, newdata = train_data)
obsT = PU_Train_noweeks$UltimateObs
predV = predict(model, newdata = Vaild_data)
obsV = PU_Vaild_noweeks$UltimateObs