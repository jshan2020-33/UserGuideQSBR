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
		#'2nearZeroVar_10P_005_NoNA',
		#'2nearZeroVar_10U_005_NoNA',
		'2nearZeroVar_P_005_NoNA',
		'2nearZeroVar_U_005_NoNA'
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
from sklearn.neural_network import MLPRegressor 
    
# Computing performance
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


def build_models(predictors, responses, predictors_vaild, responses_vaild):
    g_cv = RandomizedSearchCV(MLPRegressor(),
            param_distributions = {'hidden_layer_sizes':[(100,),(20,20)],
                                   'activation':['identity', 'logistic', 'tanh', 'relu'],
                                   'solver':['lbfgs', 'sgd', 'adam'],
                                   'alpha':[0.1,0.01,0.001,0.0001,0.00001],
                                   },        
            scoring= "neg_mean_squared_error",n_iter=50, n_jobs=3,cv=5, refit=True,random_state=22)
    predictors = num_pipeline.fit_transform(predictors)
    predictors_vaild = num_pipeline.transform(predictors_vaild)
    g_cv.fit(predictors, responses)
    model = g_cv.best_estimator_
    modelName = "ANN";
        
    model.fit(predictors, responses);
    predictions = model.predict(predictors)
    predictions_vaild = model.predict(predictors_vaild)
       
    scores = cross_val_score(model,predictors, responses,
                                 scoring="neg_mean_squared_error", cv=5,
                                 n_jobs=3)
    mse_scores = -scores
    
    MSE = mean_squared_error(responses,predictions)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(responses,predictions)
    
    MSE_vaild = mean_squared_error(responses_vaild,predictions_vaild)
    RMSE_vaild = np.sqrt(MSE_vaild)
    R2_vaild = r2_score(responses_vaild,predictions_vaild)
    
    Result = {};
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
    Result['g_cv_p'] = g_cv.best_params_
    return Result;


for nDesc in [1,2,3,4]:
    
    print('############# nDesc ' + str(nDesc) + ' Begining ############')
    Result = [] 
    
    for type in types:
        if '_P_' in type:
            Target = 'Primary Biodegradability'
            target = 'PrimaryObs'
            limmax = 5
            limmin = 2
        elif '_U_' in type:
            Target = 'Ultimate Biodegradability'
            target = 'UltimateObs'
            limmax = 4.5
            limmin = 1

        Perform = pd.read_csv(home + "StepswiseMLR/ML_Steps_" + type + ".csv")
        AddDesc = list(Perform[Perform.AddorRem=='+']['Parameter'])
        RemDesc = list(Perform[Perform.AddorRem=='x']['Parameter'])
        for i in RemDesc:
            AddDesc.remove(i)
        
        ### Extract features selected by Stepwise
        train_X = strat_train_set.loc[:,AddDesc[0:nDesc]]
        train_Y = strat_train_set[target].copy()
        vaild_X = strat_test_set.loc[:,AddDesc[0:nDesc]]
        vaild_Y = strat_test_set[target].copy()
        
        i = 10
        print('############# '+str(i) +' of  '+ target + ' in nDesc ' + str(nDesc) + ' Begining ############')
        temp = build_models(train_X,train_Y,vaild_X,vaild_Y);
        Result.append([target,temp['modelName'],nDesc,
                       temp['RMSEtrain'],temp['R2train'],
                       temp['RMSEvaild'],temp['R2vaild'],
                       np.sqrt(temp['mse_scores'].mean()),
                       temp['g_cv_p'],
                       np.sqrt(temp['mse_scores'])]);
        
    Result = pd.DataFrame(Result);
    print(Result)
    np.savetxt(home + 'Perform_'+str(nDesc) + '.csv',Result, delimiter = ',',fmt = '%s')

