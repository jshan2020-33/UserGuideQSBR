### import data

#########################################0.Remove descriptors that contain too much Na###############################
na_flag <- apply(is.na(data_ALL_X), 2, sum)
data_ALL_X_no0_5 <- data_ALL_X[, which(na_flag <= 0)]
dim(data_ALL_X_no0_5)


#########################################1.Removes descriptors whose data are all constant 0###############################
data_ALL_X_no0_5_ <- as.data.frame(lapply(data_ALL_X_no0_5,
                                          function(x) as.numeric(as.character(x))))
na_flag <- apply(is.na(data_ALL_X_no0_5_), 2, sum)
Data = data_ALL_X_no0_5_[, which(na_flag <= 0)]
dim(Data)
Data1 = Data[,which(colSums(Data,na.rm=TRUE) != 0)]
dim(Data1)


######################################## 2.Remove features that do not change###############################
nzv <- nearZeroVar(train_data, freqCut =95/5,uniqueCut = 95/5)
# 防止不存在nzv
if (length(nzv)!=0){
	filteredDescr <- train_data[, -nzv]
}else{
	filteredDescr <- train_data
}
dim(filteredDescr)


###################################### 3. Stepwise MLR ################################

filteredDescr[is.na(filteredDescr) | filteredDescr == 'Inf'] = NA
DataP = cbind(PrimaryObs = Bio[,7],filteredDescr)

library(olsrr)
fullP=lm(PrimaryObs ~ ., data=DataP)

### Save output file ###
zz <- file("ML_Steps_2nearZeroVar_P_005_NoNA.md", open = "wt")
sink(zz)
sink(zz, type = "message")

# execute steps
P_0.05_0.1_both_2nd = ols_step_both_p(fullP,pent = 0.05, prem = 0.05,details=TRUE)
P_0.05_0.1_both_2nd

# revert output back to the console -- only then access the file!
sink(type = "message")
sink()

file.show("ML_Steps_2nearZeroVar_P_005_NoNA.md")

