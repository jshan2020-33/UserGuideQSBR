
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats.stats import pearsonr

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import os


# import data
import dill
home = "D:/UserGuideQSBR/"
dill.load_session(home + 'RawData/DataSet.pkl')


#################### Specify the target and descriptor
types = [
		#'2nearZeroVar_10P_005',
		#'2nearZeroVar_10U_005',
		#'2nearZeroVar_P_005','2nearZeroVar_U_005',
		#'3highlyCor_P_005','3highlyCor_U_005',
		]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline 
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #('std_scaler', MinMaxScaler()),
    ('std_scaler', StandardScaler()),
    #("pca", PCA(n_components=0.95)),
])

# apply Regression
from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor 
    
#Computing performance
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
    
def build_models(predictors, responses, predictors_vaild, responses_vaild, modelNo):
    if(modelNo==0):
        # Linear Regression
        model = linear_model.LinearRegression();
        modelName = "Linear";
        
    if(modelNo==1):
        # Ridge Regression
        model = linear_model.RidgeCV(alphas=[0.1,1,10,100,1000]);
        modelName = "Ridge";
        predictors = num_pipeline.fit_transform(predictors)
        predictors_vaild = num_pipeline.transform(predictors_vaild)
        
    if(modelNo==2):
        # lasso Regression
        model = linear_model.LassoCV(eps=0.001, n_alphas=100,);
        modelName = "Lasso";
        predictors = num_pipeline.fit_transform(predictors)
        predictors_vaild = num_pipeline.transform(predictors_vaild)
        
    if(modelNo==3):
        # DT
        g_cv = RandomizedSearchCV(DecisionTreeRegressor(random_state=0),
                param_distributions = {'min_samples_split': range(2, 10),
                                       'max_depth':range(1,20),
                                       'min_samples_leaf':range(1,20),
                                       'min_weight_fraction_leaf':[0,0.1,0.3,0.5],
                                       'max_features':range(1,10)
                                       },        
                scoring= "neg_mean_squared_error",n_iter=50,n_jobs=3, cv=5, refit=True,random_state=22)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "DTree";

    if(modelNo==4):
        # XGBoost      
        g_cv = RandomizedSearchCV(XGBRegressor(random_state=0),
                param_distributions = {'n_estimators':range(10,510,50),
                                       'learning_rate':[0.001,0.01,0.05,0.1],
                                       'subsample':[0.5,0.7,0.9,1],
                                       'min_child_weight':range(1,40),
                                       'max_depth':range(1,20),
                                       },        
                scoring= "neg_mean_squared_error",n_iter=50, n_jobs=3,cv=5, refit=True,random_state=22)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "XGBoost";

     
    if(modelNo==5):
        # AdaBoostRegressor      
        g_cv = RandomizedSearchCV(AdaBoostRegressor(random_state=0),
                param_distributions = {'n_estimators':range(10,510,50),
                                       'learning_rate':[0.001,0.01,0.05,0.1],
                                       },        
                scoring= "neg_mean_squared_error",n_iter=50, n_jobs=3,cv=5, refit=True,random_state=22)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "AdaBoost";

    if(modelNo==6):
        # GradientBoostingRegressor    
        g_cv = RandomizedSearchCV(GradientBoostingRegressor(random_state=0),
                param_distributions = {'n_estimators':range(10,510,50),
                                       'learning_rate':[0.001,0.01,0.05,0.1],
                                       'subsample':[0.5,0.7,0.9,1],
                                       'max_depth':range(1,20),
                                       'min_samples_leaf':range(1,20),
                                       'max_leaf_nodes':range(1,20),
                                       },        
                scoring= "neg_mean_squared_error",n_iter=50, n_jobs=3,cv=5, refit=True,random_state=22)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "GBRT";

    if(modelNo==7):
        # Bagging     
        g_cv = RandomizedSearchCV(BaggingRegressor(random_state=0),
                param_distributions = {'n_estimators':range(10,510,50),
                                       'max_samples':[0.5,0.6,0.7,0.8,0.9,1],
                                       'max_features':range(1,10),
                                       },        
                scoring= "neg_mean_squared_error",n_iter=50, n_jobs=3,cv=5, refit=True,random_state=22)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "Bagging";
        
    if(modelNo==8):
        # RF      
        g_cv = RandomizedSearchCV(RandomForestRegressor(random_state=0),
                param_distributions = {'n_estimators':range(10,510,50),
                                       'max_depth':range(1,20),
                                       'max_samples':[0.5,0.7,0.9,1],
                                       'max_features':range(1,20),
                                       'min_samples_split':range(1,20),
                                       'min_samples_leaf':range(1,20),
                                       'max_leaf_nodes':range(1,20),
                                       'min_weight_fraction_leaf':[0,0.1,0.3,0.5],
                                       },        
                scoring= "neg_mean_squared_error",n_iter=50, n_jobs=3,cv=5, refit=True,random_state=22)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "RF";

    if(modelNo==9):
        # SVR      
        g_cv = RandomizedSearchCV(SVR(max_iter=100),
                param_distributions = {
                        'kernel': ['linear', 'rbf'],
                        'C': [0.1,1,5,10,100,1000,10000,100000,1000000],
                        'gamma': [0,0.1,0.5,0.01,0.001,0.0001,0.00001,0.000001,0.0000001],
                },        
                scoring= "neg_mean_squared_error",n_iter=50, n_jobs=3,cv=5, refit=True,random_state=22)
        predictors = num_pipeline.fit_transform(predictors)
        predictors_vaild = num_pipeline.transform(predictors_vaild)
        g_cv.fit(predictors, responses)
        model = g_cv.best_estimator_
        modelName = "SVR";
        
    model.fit(predictors, responses);
    predictions = model.predict(predictors)
    predictions_vaild = model.predict(predictors_vaild)
       
    scores = cross_val_score(model,predictors, responses,
                                 scoring="neg_mean_squared_error", cv=5,
                                 n_jobs=3)
    mse_scores = -scores
    
    '''
    def display_scores(scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
    display_scores(lin_rmse_scores)
    '''
    Result = {};
    if modelNo in range(3,11):
        Result['g_cv_p'] = g_cv.best_params_
        MSE = mean_squared_error(responses,predictions)
        R2 = r2_score(responses,predictions)
        MSE_vaild = mean_squared_error(responses_vaild,predictions_vaild)
        R2_vaild = r2_score(responses_vaild,predictions_vaild)
    else:
        Result['g_cv_p'] = None
        MSE = mean_squared_error(10**responses,10**predictions)
        R2 = r2_score(10**responses,10**predictions)
        MSE_vaild = mean_squared_error(10**responses_vaild,10**predictions_vaild)
        R2_vaild = r2_score(10**responses_vaild,10**predictions_vaild)

    RMSE = np.sqrt(MSE)
    RMSE_vaild = np.sqrt(MSE_vaild)    

    Result['modelName'] = modelName;
    Result['predictions'] = predictions;
    Result['predictions_vaild'] = predictions_vaild;
    Result['model'] = model;

    
    Result['MSEtrain'] = MSE
    Result['RMSEtrain'] = RMSE
    Result['R2train'] = R2
    
    Result['MSEvaild'] = MSE_vaild
    Result['RMSEvaild'] = RMSE_vaild
    Result['R2vaild'] = R2_vaild
    Result['mse_scores'] = mse_scores
    
    return Result;

for nDesc in [5,10,15,20,25,30]:

    
    print('############# nDesc ' + str(nDesc) + ' Begining ############')
    Result = [] 
    
    for type in types:
        if '_P_' in type:
            Target = 'Primary Biodegradability'
            target = 'PrimaryObs'
            limmax = 5
            limmin = 2
            mo1 = 3
            mo2 = 9
        elif '_U_' in type:
            Target = 'Ultimate Biodegradability'
            target = 'UltimateObs'
            limmax = 4.5
            limmin = 1
            mo1 = 3
            mo2 = 9            
        elif '_10P_' in type:
            Target = 'Primary Biodegradability'
            target = 'Log10P'
            limmax = 5
            limmin = 2
            mo1 = 0
            mo2 = 2            
        elif '_10U_' in type:
            Target = 'Ultimate Biodegradability'
            target = 'Log10U'
            limmax = 4.5
            limmin = 1
            mo1 = 0
            mo2 = 2            
            
        Perform = pd.read_csv(home + "StepswiseMLR/ML_Steps_" + type + ".csv")
        AddDesc = list(Perform[Perform.AddorRem=='+']['Parameter'])
        RemDesc = list(Perform[Perform.AddorRem=='x']['Parameter'])
        for i in RemDesc:
            AddDesc.remove(i)
        
        ### Extract 25 features selected by Stepwise
        train_X = strat_train_set.loc[:,AddDesc[0:nDesc]]
        train_Y = strat_train_set[target].copy()
        vaild_X = strat_test_set.loc[:,AddDesc[0:nDesc]]
        vaild_Y = strat_test_set[target].copy()
    
        for i in range(mo1,mo2+1):
            print('############# '+str(i) +' of  '+ target + ' in nDesc ' + str(nDesc) + ' Begining ############')
            temp = build_models(train_X,train_Y,vaild_X,vaild_Y, i);
            Result.append([target,temp['modelName'],nDesc,
                           temp['RMSEtrain'],temp['R2train'],
                           temp['RMSEvaild'],temp['R2vaild'],
                           np.sqrt(temp['mse_scores'].mean()),
                           temp['g_cv_p'],
                           np.sqrt(temp['mse_scores'])]);
        
    Result = pd.DataFrame(Result);
    print(Result)
    np.savetxt(home + 'Ridge/Perform_'+str(nDesc) + '.csv',Result, delimiter = ',',fmt = '%s')        

