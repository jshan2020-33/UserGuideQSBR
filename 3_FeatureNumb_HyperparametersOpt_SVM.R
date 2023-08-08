graphics.off()
rm(list = ls(all=TRUE))

setwd("D:/UserGuideQSBR")
load("/RawData/DataSet.RData")

home = "D:/UserGuideQSBR/"

library(tidyverse)
library(caTools)
library(caret)
library(e1071)
library(GA)


types = c(
		#'2nearZeroVar_10P_005_NoNA',
		#'2nearZeroVar_10U_005_NoNA',
		'2nearZeroVar_P_005_NoNA',
		'2nearZeroVar_U_005_NoNA'
		)


## Normalized features

# For primary biodegradatio rate
MaxminScacleP = function(PU_Train_noweeks,PU_Vaild_noweeks){

	center = sweep(PU_Train_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
	R = apply(PU_Train_noweeks[,-1],2,max) - apply(PU_Train_noweeks[,-1],2,min)
	data_x_01_T = sweep(center,2,R,"/")
	Train_data = cbind(PrimaryObs=PU_Train_noweeks$PrimaryObs,data_x_01_T)

	center_V = sweep(PU_Vaild_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
	data_x_01_V = sweep(center_V,2,R,"/")
	Vaild_data = cbind(PrimaryObs=PU_Vaild_noweeks$PrimaryObs,data_x_01_V)

	return(list(Train_data,Vaild_data))


}

# For ultimate biodegradatio rate
MaxminScacleU = function(PU_Train_noweeks,PU_Vaild_noweeks){

	center = sweep(PU_Train_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
	R = apply(PU_Train_noweeks[,-1],2,max) - apply(PU_Train_noweeks[,-1],2,min)
	data_x_01_T = sweep(center,2,R,"/")
	Train_data = cbind(UltimateObs=PU_Train_noweeks$UltimateObs,data_x_01_T)

	center_V = sweep(PU_Vaild_noweeks[,-1],2,apply(PU_Train_noweeks[,-1],2,min),"-")
	data_x_01_V = sweep(center_V,2,R,"/")
	Vaild_data = cbind(UltimateObs=PU_Vaild_noweeks$UltimateObs,data_x_01_V)

	return(list(Train_data,Vaild_data))
}


for (type in types){
	
	Perform = read.csv(file=paste(home,"StepswiseMLR/ML_Steps_",type, ".csv", sep=""),head = TRUE)
	
	AddDesc = Perform[Perform$AddorRem=='+',]$Parameter
	RemDesc = Perform[Perform$AddorRem=='x',]$Parameter
	
	## remove RemDesc in AddDesc
	Desc = AddDesc # In order not to destroy AddDesc

	for(i in RemDesc){Desc = Desc[-which(Desc==i)]} # Loop to delete variables in RemDesc one by one

	if (grepl('P',type)){target = 'PrimaryObs'}else{target = 'UltimateObs'}

	ker = 'radial'
	#ker = 'linear'
	nDescnum = c(10:length(Desc))
	df=data.frame(nDesc = nDescnum)
	
	df$nSample = NA
	df$Target = target
	df$Kernel = ker
	df$Cost = NA
	df$Gamma = NA
	Process = 'Maxmin01'
	df$Procesing = Process
	
	df$RMSEtrain = NA
	df$R2train = NA
	df$MAEtrain = NA # Mean Absolute Error
	
	df$RMSEvaild = NA
	df$Q2vaild = NA

	df$RMSEtest = NA
	df$R2test = NA
	df$MAEtest = NA # Mean Absolute Error

	df$RMSEtotal = NA
	df$R2total = NA
	df$MAEtotal = NA # Mean Absolute Error

    print(paste("##########",type," Begining","##########"))
	for (i in c(10:length(Desc))){
	
		print(paste("#####",i," in ",type," Begining","#####"))
		
		### Define data set
		Data = data_ALL[,c(target,Desc[1:i])]
		set.seed(22)
		TM=sample.split(Data[,1],4/5)
		T_data=Data[TM,]
		V_data=Data[!TM,]
		
		if (grepl('P',type)){
			MaxminData = MaxminScacleP(T_data,V_data)
		}else{
			MaxminData = MaxminScacleU(T_data,V_data)
		}
		train_data = MaxminData[[1]]
		vaild_data = MaxminData[[2]]

		### Define the tuning function
		evalParams <- function(x,Train_data) {

			set.seed(22)
			K = 10 # 10-fold cross-validation 
			fold_inds <- sample(1:K, nrow(Train_data), replace = TRUE)

			pred = c(1:nrow(Train_data))

			for(j in 1:K){
				training = Train_data[fold_inds != j, , drop = FALSE]
				# Train 
				if (grepl('P',type)){
					set.seed(22)
					model <- svm( PrimaryObs~., data = training, type = "eps-regression", cost=x[1],gamma=x[2],kernel = ker,scale=0)#radial
					obs=Train_data$PrimaryObs
				}else{
					set.seed(22)
					model <- svm( UltimateObs~., data = training, type = "eps-regression", cost=x[1],gamma=x[2],kernel = ker,scale=0)#radial
					obs=Train_data$UltimateObs
				}
				validation = Train_data[fold_inds == j, , drop = FALSE]
				pred[fold_inds == j] = predict(model, validation)
			}

			RMSE = postResample(pred,obs)[[1]]
			R2 = postResample(pred,obs)[[2]]

			return(-RMSE)
		} 
		
		### Define the scope of the parameters
		theta_min <- c(cost = 1, gamma = 1e-6) 
		theta_max <- c(cost = 10000, gamma = 1)
		#theta_min <- c(cost = 1, gamma = 1e-8) 
		#theta_max <- c(cost = 200, gamma = 1e-2)

		### Perform GA parameter optimization
		results <- ga(fitness =function(x) evalParams(x=x,Train_data=train_data),
					  type = "real-valued",
					  names = names(theta_min), 
					  lower = theta_min, upper = theta_max, 
					  popSize = 60, maxiter = 1000,
					  run = 100,
					  parallel = T)
		
		### GA optimal model calculation
		if (grepl('P',type)){
			model=svm( PrimaryObs~., data = train_data,
						type = "eps-regression",
						cost=results@solution[1,1],gamma=results@solution[1,2],
						kernel = ker,scale=0)
			obsT=train_data$PrimaryObs
			obsV=vaild_data$PrimaryObs
			lim_min = 2
			lim_max = 5
			lim = 0.3
			at = c(2.0,2.5,3.0,3.5,4.0,4.5,5.0)			
			Target = "Primary Biodegradability"			
		}else{
			model=svm( UltimateObs~., data = train_data,
						type = "eps-regression",
						cost=results@solution[1,1],gamma=results@solution[1,2],
						kernel = ker,scale=0)
			obsT=train_data$UltimateObs
			obsV=vaild_data$UltimateObs
			lim_min = 1
			lim_max = 4.5
			lim = 0.35
			at = c(1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5)			
			Target = "Ultimate Biodegradability"			
		}

		predT=predict(model, newdata = train_data)
		predV=predict(model, newdata = vaild_data)

		############# Record nDesc performance   		
		rindx=grep(paste("^",i,"$",sep=""),df$nDesc)
		df$Cost[rindx] = results@solution[1,1]
		df$Gamma[rindx] = results@solution[1,2]
	
		postResample(predT,obsT)
		postResample(predV,obsV)
		postResample(c(predT,predV),c(obsT,obsV))

		df$RMSEtrain[rindx] = postResample(predT,obsT)[1]
		df$R2train[rindx] = postResample(predT,obsT)[2]
		df$MAEtrain[rindx] = postResample(predT,obsT)[3] # Mean Absolute Error

		df$RMSEtest[rindx] = postResample(predV,obsV)[1]
		df$R2test[rindx] = postResample(predV,obsV)[2]
		df$MAEtest[rindx] = postResample(predV,obsV)[3] # Mean Absolute Error

		df$RMSEtotal[rindx] = postResample(c(predT,predV),c(obsT,obsV))[1]
		df$R2total[rindx] = postResample(c(predT,predV),c(obsT,obsV))[2]
		df$MAEtotal[rindx] = postResample(c(predT,predV),c(obsT,obsV))[3] # Mean Absolute Error
		
		df$nSample[rindx] = dim(Data)[1]
		
		# Add a verification
		pred = list()

		for(i in 1:nrow(train_data)){
			training = train_data[-i,]
			# Train 
			if (grepl('P',type)){
				model <- svm( PrimaryObs~., data = training, type = "eps-regression",cost=results@solution[1,1],gamma=results@solution[1,2],kernel = ker,scale=0)#radial
				obs = train_data$PrimaryObs
			}else{
				model <- svm( UltimateObs~., data = training, type = "eps-regression",cost=results@solution[1,1],gamma=results@solution[1,2],kernel = ker,scale=0)#radial
				obs = train_data$UltimateObs
			}
			validation = train_data[i,]
			pred[[i]] = predict(model, validation)
		}
		
		df$RMSEvaild[rindx] = postResample(unlist(pred),obs)[[1]]
		df$Q2vaild[rindx] = postResample(unlist(pred),obs)[[2]]

		write.csv(df,
		file=paste(home,"SVM_MLSteps_DescChange_10KFold_",
		target,"_",ker,"_",Process,".csv",sep = ""),row.names=F)
	}
}
