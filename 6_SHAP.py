### Description
# SHAP analysis takes SVM model as an example,
# and different model objects can be replaced by actual operation

import pandas as pd
import numpy as np
import os


### Define preprocessing function
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', MinMaxScaler()),
    #('std_scaler', StandardScaler()),
    #("pca", PCA(n_components=0.95)),
])


### import data
import dill
home = "D:/UserGuideQSBR/"
dill.load_session(home + 'RawData/DataSet.pkl')

home = "D:/UserGuideQSBR/"


# Model implementation
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Calculat performance
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import shap


### Set up the content related to the ultimate SVM model
al = "SVM"
model = SVR(kernel = 'linear',
            C = 21.59313785,
            )
Target = 'Ultimate Biodegradability'
target = 'UltimateObs'
limmax = 4.5
limmin = 1
nDesc = 44
type = '2nearZeroVar_U_005_NoNA'
Perform = pd.read_csv(home + "StepswiseMLR/ML_Steps_" + type + ".csv")
AddDesc = list(Perform[Perform.AddorRem=='+']['Parameter'])
RemDesc = list(Perform[Perform.AddorRem=='x']['Parameter'])
for i in RemDesc:
    AddDesc.remove(i)
Desc = AddDesc[0:nDesc]

### Extract features selected by Stepwise
train_X = strat_train_set.loc[:,Desc]
train_Y = strat_train_set[target].copy()
test_X = strat_test_set.loc[:,Desc]
test_Y = strat_test_set[target].copy()

### Data preprocessing
train_X = pd.DataFrame(num_pipeline.fit_transform(train_X),columns = Desc)
test_X = pd.DataFrame(num_pipeline.transform(test_X),columns = Desc)
    
model.fit(train_X, train_Y)

################ based on Training
background = shap.maskers.Independent(train_X)
### establish Explainer    
explainer = shap.KernelExplainer(model = model.predict,
                                 data= shap.kmeans(train_X,5),
                                masker = background,
                                 )
# Using 138 background data samples could cause slower run times.
# Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.

shap_values = explainer.shap_values(train_X)
shap_values2 = shap.Explanation(shap_values,
                               data=train_X,
                               base_values = np.array([explainer.expected_value]*35),
                               feature_names = Desc
                               )

shap.plots.beeswarm(shap_values2, 
                    order=shap_values2.abs.mean(0))

################ based on Testing

shap_values = explainer.shap_values(test_X)
shap_values2 = shap.Explanation(shap_values,
                               data=test_X,
                               base_values= np.array([explainer.expected_value]*35),
                               feature_names=Desc
                               )

shap.plots.beeswarm(shap_values2, 
                    order=shap_values2.abs.mean(0))


################ Visualization of a single sample predictive interpretation

### 1、force plot
shap.force_plot(explainer.expected_value, 
                shap_values[1,:], 
                test_X.iloc[1,:],
                matplotlib=True)
### 2、Decision plot
shap.decision_plot(explainer.expected_value,
                   shap_values[1,:],
                   test_X.iloc[1,:])
### 3、Waterfall plot
shap.plots.waterfall(shap_values2[1])
