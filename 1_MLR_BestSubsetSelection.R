setwd("D:/UserGuideQSBR") # Setting working directory
home = getwd() # Getting current path

load("/RawData/DataSet.RData")
source("/Function_Allsubset_Log10P.R")
source("/Function_Allsubset_Log10U.R")

library(tidyverse)
data_ALL_Self = select(data_ALL,c(E
								 ,V_McGowanVolume
								 ,MolecularWeight
								 ,No..of.Rotatable.Bonds
								 ,N.Ratio
								 ,Halogen.Ratio
								 ,Number.of.Rings..size.6.
								 ,Index.of.Refraction
								 ,Polarizability
								 ,Molar.Refractivity
								 ,ESPI
								 ,TDE0
								 ,Hardness
								 ,Number.of.Rings
								 ,Parachor
								 ,Density
								 ,Entropy
								 ,Volume
								 ,Density
								 ,ESPA
								 ,DEmax
								 ,logP_Soil
								 ,No..of.Hydrogen.Bond.Acceptors
								 ,NO.Ratio
								 ,Molar.Volume
								 ,LUMO
								 ,Nvariance
								 ,LccMin
								 ,LccAver
								 ,DEmin
								 ,DE0max
								 ,TDE
								 ,FEDLUMOmax
								 ,S
								 ,A
								 ,C.Ratio
								 ,Hetero.Ratio
								 ,logSw
								 ,MELTING_POINT_DEGC_OPERA_PRED
								 ,Polarizability
								 ,HLG
								 ,PA
								 ,N_A
								 ,ESPPA
								 ,ESPNA
								 ,Pi
								 ,MPI
								 ,NSA
								 ,PSA
								 ,LccMax
								 ,DE0min
								 ,TDN
								 ,FwHOMOmin
								 ,FwRadicalmin
								 ,FHOMOmin
								 ,ConElectrophilicitymin))

data_ALL_Steps = select(data_ALL,c(GraphFP245
									 ,PatternFP695_RDKit
									 ,C1SP3
									 ,AATS0v
									 ,AATS3p
									 ,MDEO.12
									 ,Ks_R
									 ,GraphFP212
									 ,MHFP186_RDKit
									 ,ExtFP594
									 ,GATS7s
									 ,ecfp8_1379_open
									 ,MHFP1493_RDKit
									 ,PubchemFP3
									 ,RDKFP933_RDKit
									 ,MHFP1299_RDKit
									 ,RDKFP1237_RDKit
									 ,ExtFP858
									 ,AD2D237
									 ,RDKFP610_RDKit
									 ,RDKFP178_RDKit
									 ,ExtFP15
									 ,MHFP1142_RDKit
									 ,MHFP448_RDKit
									 ,RDKFP798_RDKit
									 ,RDKFP1392_RDKit
									 ,ExtFP161
									 ,FP350
									 ,FP895
									 ,MHFP1343_RDKit
									 ,RDKFP112_RDKit
									 ,ECFPs1024_716_RDKit
									 ,MHFP260_RDKit
									 ,PatternFP548_RDKit
									 ,MHFP1210_RDKit
									 ,MHFP1552_RDKit
									 ,MHFP1601_RDKit
									 ,RDKFP758_RDKit
									 ,PatternFP1967_RDKit
									 ,RDKFP241_RDKit
									 ,MHFP1048_RDKit
									 ,MHFP1661_RDKit
									 ,MHFP1009_RDKit
									 ,RDKFP192_RDKit
									 ,RDKFP595_RDKit
									 ,FP2_8_open.1
									 ,RDKFP287_RDKit
									 ,RDKFP1299_RDKit
									 ,RDKFP1711_RDKit
									 ,MHFP1274_RDKit
									 ,Ai_R
									 ,RDKFP342_RDKit))

target             <- c("Log10P","Log10U")
nDesc              <- seq(8,15, by = 1)

grid_df          <- expand.grid(target,nDesc)
names(grid_df)   <- c("target", "nDesc")

############################ Parallel method [Run successfully]
library(foreach)
library(doParallel)

cl            <- makeCluster(24)
registerDoParallel(cl)

start_proc    <- Sys.time()


results <- foreach(x = 1:nrow(grid_df),
					.combine=rbind,
					.packages = c('leaps','caret','caTools'),
					.verbose=TRUE
					) %dopar% {
  
	start_task <- Sys.time()

	if (grepl('P',grid_df$target[x])){
		train_data = cbind(Log10P = data_ALL$Log10P, data_ALL_Self, data_ALL_Steps)
		allsubsetsP=regsubsets(Log10P~.,data = train_data, nbest=5, nvmax=grid_df$nDesc[x],method='exhaustive',really.big=TRUE)
		save.image(paste("1_LM_",grid_df$target[x],"_5",grid_df$nDesc[x],".RData",sep = ""))
		Function_Allsubset_Log10P()
	}else{
		train_data = cbind(Log10U = data_ALL$Log10U, data_ALL_Self, data_ALL_Steps)
		allsubsetsP=regsubsets(Log10U~.,data = train_data, nbest=5, nvmax=grid_df$nDesc[x],method='exhaustive',really.big=TRUE)
		save.image(paste("1_LM_",grid_df$target[x],"_5",grid_df$nDesc[x],".RData",sep = ""))
		Function_Allsubset_Log10U()
	}
	
	end_task <- Sys.time()
	
	return(c(grid_df$nDesc[x],grid_df$target[x],end_task - start_task))

}

stopCluster(cl)
end_proc <- Sys.time()
end_proc - start_proc

write.csv(results, file="nDesc_Allsubset_TimeSpent.csv", row.names=F)