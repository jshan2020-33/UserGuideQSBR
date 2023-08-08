############ 不同特征个数的Best性能计算
# 存储5*8个模型描述符选择情况
Function_Allsubset_Log10U = function(){

	library(leaps)
	a = summary(allsubsetsP)$which

	library(caTools)
	set.seed(22)
	TM=sample.split(train_data$Log10U,4/5) #不能对矩阵用$提取对应名称的列
	T_data = train_data[TM,]
	V_data = train_data[!TM,]

	Model = c(1:dim(a)[1])
	df=data.frame(Models = Model)

	df$Target = NA

	df$RMSEtrain = NA
	df$R2tr = NA
	df$MAEtr = NA # Mean Absolute Error

	df$RMSEtest = NA
	df$R2test = NA
	df$MAEtest = NA # Mean Absolute Error

	df$RMSEtotal = NA
	df$R2total = NA
	df$MAEtotal = NA # Mean Absolute Error

	df$nSample = NA
	df$nDesc = NA

	df$Intercept = NA
	df$coef1 = NA
	df$Parameter1 = NA
	df$coef2 = NA
	df$Parameter2 = NA
	df$coef3 = NA
	df$Parameter3 = NA
	df$coef4 = NA
	df$Parameter4 = NA
	df$coef5 = NA
	df$Parameter5 = NA
	df$coef6 = NA
	df$Parameter6 = NA
	df$coef7 = NA
	df$Parameter7 = NA
	df$coef8 = NA
	df$Parameter8 = NA
	df$coef9 = NA
	df$Parameter9 = NA
	df$coef10 = NA
	df$Parameter10 = NA
	df$coef11 = NA
	df$Parameter11 = NA
	df$coef12 = NA
	df$Parameter12 = NA

	library(caret)
	for (i in Model){
		
		# 存储单个模型描述符选择情况
		b = a[i,]
		train_data = T_data[,b]
		test_data = V_data[,b]
		
		LM = lm(Log10U~.,data=train_data)
		
		obsTrain=10^train_data$Log10U
		predTrain=10^predict(LM,train_data)
		
		obsTest=10^test_data$Log10U
		predTest=10^predict(LM,test_data)
		
		rindx=grep(paste("^",i,"$",sep=""),df$Models)
		
		Name = names(train_data)
		
		df$Target[rindx] = Name[1]

		df$RMSEtrain[rindx] = postResample(predTrain,obsTrain)[1]
		df$R2tr[rindx] = postResample(predTrain,obsTrain)[2]
		df$MAEtr[rindx] = postResample(predTrain,obsTrain)[3] # Mean Absolute Error

		df$RMSEtest[rindx] = postResample(predTest,obsTest)[1]
		df$R2test[rindx] = postResample(predTest,obsTest)[2]
		df$MAEtest[rindx] = postResample(predTest,obsTest)[3] # Mean Absolute Error

		df$RMSEtotal[rindx] = postResample(c(predTrain,predTest),c(obsTrain,obsTest))[1]
		df$R2total[rindx] = postResample(c(predTrain,predTest),c(obsTrain,obsTest))[2]
		df$MAEtotal[rindx] = postResample(c(predTrain,predTest),c(obsTrain,obsTest))[3] # Mean Absolute Error

		df$nSample[rindx] = dim(na.omit(train_data))[1] + dim(na.omit(test_data))[1]
		nDesc_num = dim(na.omit(train_data))[2]-1
		df$nDesc[rindx] = nDesc_num

		df$Intercept[rindx] = LM$coefficients[1]
		df$coef1[rindx] = LM$coefficients[2]
		df$Parameter1[rindx] = names(LM$coefficients[2])
		if (nDesc_num == 1){next}
		
		df$coef2[rindx] = LM$coefficients[3]
		df$Parameter2[rindx] = names(LM$coefficients[3])
		if (nDesc_num == 2){next}
		
		df$coef3[rindx] = LM$coefficients[4]
		df$Parameter3[rindx] = names(LM$coefficients[4])
		if (nDesc_num == 3){next}
		
		df$coef4[rindx] = LM$coefficients[5]
		df$Parameter4[rindx] = names(LM$coefficients[5])
		if (nDesc_num == 4){next}
		
		df$coef5[rindx] = LM$coefficients[6]
		df$Parameter5[rindx] = names(LM$coefficients[6])
		if (nDesc_num == 5){next}
		
		df$coef6[rindx] = LM$coefficients[7]
		df$Parameter6[rindx] = names(LM$coefficients[7])
		if (nDesc_num == 6){next}
		
		df$coef7[rindx] = LM$coefficients[8]
		df$Parameter7[rindx] = names(LM$coefficients[8])
		if (nDesc_num == 7){next}
		
		df$coef8[rindx] = LM$coefficients[9]
		df$Parameter8[rindx] = names(LM$coefficients[9])
		if (nDesc_num == 8){next}
		
		df$coef9[rindx] = LM$coefficients[10]
		df$Parameter9[rindx] = names(LM$coefficients[10])
		if (nDesc_num == 9){next}
		
		df$coef10[rindx] = LM$coefficients[11]
		df$Parameter10[rindx] = names(LM$coefficients[11])
		if (nDesc_num == 10){next}
		
		df$coef11[rindx] = LM$coefficients[12]
		df$Parameter11[rindx] = names(LM$coefficients[12])
		if (nDesc_num == 11){next}
		
		df$coef12[rindx] = LM$coefficients[13]
		df$Parameter12[rindx] = names(LM$coefficients[13])

	}
	write.csv(df,
			file=paste("0 ML_Steps_2nearZeroVar_10U_005_resub_",
			length(Model),".csv",sep = ""),row.names=F)
	print(paste("##### 0 ML_Steps_2nearZeroVar_10U_005_resub_",length(Model),".csv Done",sep = ""))

}