setwd("D:/UserGuideQSBR")
load("/RawData/DataSet.RData")

home = "D:/UserGuideQSBR/"

library(e1071)
library(xgboost)
library(caret)
library(Matrix)

library(caTools)
set.seed(22)
TM=sample.split(data_ALL$Log10P,4/5) 
T_data=data_ALL[TM,]
V_data=data_ALL[!TM,]

types = c(
		'2nearZeroVar_P_005_NoNA',
		'2nearZeroVar_U_005_NoNA'
		)

for (type in types){

	Perform = read.csv(file=paste(home,"StepswiseMLR/ML_Steps_",type, ".csv", sep=""),head = TRUE)

	AddDesc = Perform[Perform$AddorRem=='+',]$Parameter
	RemDesc = Perform[Perform$AddorRem=='x',]$Parameter

	## remove RemDesc in AddDesc
	Desc = AddDesc # In order not to destroy AddDesc
	for(i in RemDesc){Desc = Desc[-which(Desc==i)]} # Loop to delete variables in RemDesc one by one

	if (grepl('P',type)){target = 'PrimaryObs'}else{target = 'UltimateObs'}

	######################### Optimize grid Settings
	nrounds = 500
	nthreads = 7
	max_depth_max = 13
	max_depth_steps = 2
	subsample_min = 0.25
	subsample_max = 1
	subsample_steps = 0.5
	colsample_bytree_min = 1
	colsample_bytree_max = 1
	colsample_bytree_steps = 0.3
	colsample_bylevel_min = 0.4
	colsample_bylevel_max = 1
	colsample_bylevel_steps = 0.6
	lambda_min = 0
	lambda_max = 0
	lambda_steps = 2


	max_depth         <- seq(1,max_depth_max, by = max_depth_steps)
	eta               <- c(0.009,0.01,0.05,0.1,0.3)
	gamma             <- c(0,0.01)
	min_child_weight  <- c(1,3,5)
	subsample         <- seq(subsample_min,subsample_max, by = subsample_steps) 
	colsample_bylevel <- seq(colsample_bylevel_min,colsample_bylevel_max, by = colsample_bylevel_steps) 
	colsample_bytree <- seq(colsample_bytree_min,colsample_bytree_max, by = colsample_bytree_steps)
	lambda            <- seq(lambda_min,lambda_max, by = lambda_steps)
	grid_df          <- expand.grid(lambda,max_depth,eta,gamma,min_child_weight,subsample,colsample_bytree,colsample_bylevel)
	names(grid_df)   <- c('lambda','max_depth','eta','gamma','min_child_weight','subsample','colsample_bytree','colsample_bylevel')
	
	######################### Set data frame
	nDescnum = c(1:length(Desc))
	df=data.frame(nDesc = nDescnum)
	
	df$nSample = NA
	df$Target = target

	df$nround = nrounds
	
	df$lambda = NA
	df$lambda_min = lambda_min
	df$lambda_max = lambda_max

	df$max_depth = NA
	df$max_depth_min = 1
	df$max_depth_max = max_depth_max

	df$eta = NA
	df$eta_min = 0.009
	df$eta_max = 0.3

	df$gamma = NA
	df$gamma_min = 0
	df$gamma_max = 0.01

	df$min_child_weight = NA
	df$min_child_weight_min = 1
	df$min_child_weight_max = 5

	df$subsample = NA
	df$subsample_min = subsample_min
	df$subsample_max = subsample_max

	df$colsample_bytree = NA
	df$colsample_bytree_min = colsample_bytree_min
	df$colsample_bytree_max = colsample_bytree_max

	df$colsample_bylevel = NA
	df$colsample_bylevel_min = colsample_bylevel_min
	df$colsample_bylevel_max = colsample_bylevel_max

	Process = '-'
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

	df$nBest = NA #The number of optimal results

	print(paste("##########",type," Begining","##########"))
	for (i in c(2:30)){
		
		print(paste("#####",i," in ",type," Begining","#####"))

		Train_data = T_data[,c(target,Desc[1:i])]
		Vaild_data = V_data[,c(target,Desc[1:i])]

		# Transform the independent variable into a sparse matrix
		if (grepl('P',type)){
			train_matrix <- sparse.model.matrix(PrimaryObs ~ .-1, data = Train_data)
			test_matrix <- sparse.model.matrix(PrimaryObs ~ .-1, data = Vaild_data)
			obs=Train_data$PrimaryObs
		}else{
			train_matrix <- sparse.model.matrix(UltimateObs ~ .-1, data = Train_data)
			test_matrix <- sparse.model.matrix(UltimateObs ~ .-1, data = Vaild_data)
			obs=Train_data$UltimateObs
		}

		# Merge the independent and dependent variables into a list
		train_fin <- list(data=train_matrix,label=Train_data[,1]) 
		test_fin <- list(data=test_matrix,label=Vaild_data[,1]) 

		dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label) 
		dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)
		
		############################ Parallel method 1 [Run successfully]
		library(foreach)
		library(doParallel)

		cl            <- makeCluster(nthreads)
		registerDoParallel(cl)

		start_proc    <- Sys.time()


		results <- foreach(x = 1:nrow(grid_df),
							.combine = rbind,
							.packages = c('xgboost','caret','caTools','Matrix'),
							.verbose = FALSE) %dopar% {
		  
			start_task <- Sys.time()
			set.seed(22)
			K = 10 # 10-fold cross-validation 
			fold_inds <- sample(1:K, nrow(Train_data), replace = TRUE)

			pred = c(1:nrow(Train_data))

			for(j in 1:K){
				training = xgb.DMatrix(data = train_fin$data[fold_inds != j, , drop = FALSE],
									   label = train_fin$label[fold_inds != j,drop = FALSE])
				# Train 
				set.seed(22)
				model <- xgboost(data = training, nround=nrounds,
						lambda = grid_df$lambda[x],
						max_depth = grid_df$max_depth[x],
						eta = grid_df$eta[x],
						gamma = grid_df$gamma[x],
						min_child_weight = grid_df$min_child_weight[x],
						subsample = grid_df$subsample[x],
						colsample_bytree = grid_df$colsample_bytree[x],
						colsample_bylevel = grid_df$colsample_bylevel[x])
				validation = xgb.DMatrix(data = train_fin$data[fold_inds == j, , drop = FALSE],
									   label = train_fin$label[fold_inds == j, drop = FALSE])
				pred[fold_inds == j] = predict(model, validation)
			}

			RMSE = postResample(pred,obs)[[1]]
			R2 = postResample(pred,obs)[[2]]
			
			end_task <- Sys.time()
			
			print(paste("###Tast spend",end_task - start_task,"###",sep = "  "))
			
			return(c(x,target,grid_df$lambda[x],
							  grid_df$max_depth[x],grid_df$eta[x],
							  grid_df$gamma[x],grid_df$min_child_weight[x],
							  grid_df$subsample[x],grid_df$colsample_bytree[x],
							  grid_df$colsample_bylevel[x],RMSE,R2))
		}
		stopCluster(cl)
		end_proc <- Sys.time()
		print(paste("#####",i," in ",type," end, Spend Time",end_proc - start_proc,"#####"))
		
		write.csv(results, file=paste(home,"XGB_10KFold_",target,"_nDesc",i,".csv",sep = ""), row.names=F)

		Best = results[which(results[,11] == min(results[,11])),]
		
		nBest = length(Best)/12
		if(nBest!=1){
		Best = Best[1,]
		}
		
		set.seed(22)
		model <- xgboost(data = dtrain, nround=nrounds,nthread = nthreads,
			lambda = as.numeric(Best[3]),
			max_depth = as.numeric(Best[4]),
			eta = as.numeric(Best[5]),
			gamma = as.numeric(Best[6]),
			min_child_weight = as.numeric(Best[7]),
			subsample = as.numeric(Best[8]),
			colsample_bytree = as.numeric(Best[9]),
			colsample_bylevel = as.numeric(Best[10]))
				
		if (grepl('P',type)){
			obsT=Train_data$PrimaryObs
			obsV=Vaild_data$PrimaryObs
			lim_min = 2
			lim_max = 5
			lim = 0.3
			at = c(2.0,2.5,3.0,3.5,4.0,4.5,5.0)			
			Target = "Primary Biodegradability"
		}else{
			obsT=Train_data$UltimateObs
			obsV=Vaild_data$UltimateObs
			lim_min = 1
			lim_max = 4.5
			lim = 0.35
			at = c(1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5)			
			Target = "Ultimate Biodegradability"
		}
		predT=predict(model, newdata = dtrain)
		predV=predict(model, newdata = dtest)

			   xpd = T,bty="n",cex=3)

		#############  Record nDesc performance  
		rindx=grep(paste("^",i,"$",sep=""),df$nDesc)
		df$nBest[rindx] = nBest
		df$lambda[rindx] = as.numeric(Best[3])
		df$max_depth[rindx] = as.numeric(Best[4])
		df$eta[rindx] = as.numeric(Best[5])
		df$gamma[rindx] = as.numeric(Best[6])
		df$min_child_weight[rindx] = as.numeric(Best[7])
		df$subsample[rindx] = as.numeric(Best[8])
		df$colsample_bytree[rindx] = as.numeric(Best[9])
		df$colsample_bylevel[rindx] = as.numeric(Best[10])

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
		
		df$nSample[rindx] = dim(na.omit(Train_data))[1] + dim(na.omit(Vaild_data))[1]

		# Add a verification
		set.seed(22)
		K = nrow(Train_data) # K-fold cross-validation 
		fold_inds <- sample(1:K, nrow(Train_data), replace = TRUE)

		pred = c(1:nrow(Train_data))

		for(j in 1:K){
			training = xgb.DMatrix(data = train_fin$data[fold_inds != j, , drop = FALSE],
								   label = train_fin$label[fold_inds != j,drop = FALSE])
			# Train 
			set.seed(22)
			model <- xgboost(data = training, nround=nrounds,nthread = nthreads,
				lambda = as.numeric(Best[3]),
				max_depth = as.numeric(Best[4]),
				eta = as.numeric(Best[5]),
				gamma = as.numeric(Best[6]),
				min_child_weight = as.numeric(Best[7]),
				subsample = as.numeric(Best[8]),
				colsample_bytree = as.numeric(Best[9]),
				colsample_bylevel = as.numeric(Best[10]))
					#nthread = 15
			validation = xgb.DMatrix(data = train_fin$data[fold_inds == j, , drop = FALSE],
								   label = train_fin$label[fold_inds == j, drop = FALSE])
			pred[fold_inds == j] = predict(model, validation)
		}

		df$RMSEvaild[rindx] = postResample(pred,obs)[[1]]
		df$Q2vaild[rindx] = postResample(pred,obs)[[2]]

		write.csv(df,
		file=paste(home,"XGB_MLSteps_DescChange_10KFold_",
		target,".csv",sep = ""),row.names=F)
	}
	dev.off()
}

