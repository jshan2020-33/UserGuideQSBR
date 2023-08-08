### User guide of QSBR models

## Folder content description
RawData: DataSet.RData (for R) and DataSet.pkl (for python)
StepswiseMLR: features pre-screened by Stepwise MLR
Unifed_XGBoost_hyperparameters: the optimized hyperparameters by grid search for the unifed XGBoost model


## Feature Prescreening（include 3 steps）
0_FeatureCleaning.R


## best subset selection for MLR
1_MLR_BestSubsetSelection.R
Function_Allsubset_Log10P.R
Function_Allsubset_Log10U.R


## Pre-screening of machine learning algorithms
2_Prescreening_ML_Others.py
2_Prescreening_ML_ANN.py


## Machine learning models develop & hyperparameters optimize
3_FeatureNumb_HyperparametersOpt_SVM.R
3_FeatureNumb_HyperparametersOpt_XGB.R


## The final model

# models for primary or ultimate biodegradation endpoints
4_ModelDevelpment_Primary_Ultimate.R
# unified biodegradation model by random sample
4_ModelDevelpment_Unified.R


## Calculating statistical parameters for model evaluation
5_Model_evaluation.R


## Shap analysis
6_SHAP.py
