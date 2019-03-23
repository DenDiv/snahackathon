from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_auc_score
import pandas as pd
import pyarrow.parquet as parquet
# Used to train the baseline model
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from mult_test import *

def regr(col_name:str):
    train_dates = [[('date', '=', '2018-03-' + x)] for x in ['12']]
    train = parquet.ParquetDataset(
        # Path to the dataset
        input_path + '/collabTrain/',
        # Dates to read
        filters = train_dates)\
        .read(
        # Columns to read
        columns = col_to_read).to_pandas()

    train =  preproc_data(train[col_of_int])

    #test = parquet.read_table(input_path + '/collabTest').to_pandas()
    test = parquet.read_table(input_path + '/collabTrain/date=2018-03-19').to_pandas()
    test =  preproc_data(test[col_of_int])
    #df_2 = feat_fr_fb_train(test['feedback'])
    #test.drop(['feedback'],axis=1,inplace=True)


    #test = pd.concat([test,df_2],axis=1)
    #y = test[test[col_name].isna()==False][col_name].values
    #X = test[test[col_name].isna()==False]
    #X = X.fillna(0.0).drop([col_name],axis=1).values
    #X_train, X_test, y_train, y_test = train_test_split(\
        #X, y, test_size=0.2, random_state=42)
    y_train = train[train[col_name].isna()==False][col_name].values
    X_train = train[train[col_name].isna()==False].drop([col_name],axis=1).fillna(0.0)
    del train
    y_test = test[test[col_name].isna()==False][col_name].values
    X_test = test[test[col_name].isna()==False].drop([col_name],axis=1).fillna(0.0)
    del test
    reg = xgboost.XGBRegressor(nthread=4).fit(X_train.fillna(0.0),y_train)
    #reg = LinearRegression().fit(X_train,y_train)
    #xgboost.plot_importance(reg)
    #plt.show()
    #del X, y
    #for ind, row in test.iterrows():
        #if np.isnan(row[col_name]):
            #test.loc[ind,col_name] = reg.predict([test.loc[ind].drop(col_name).fillna(0.0).values])

    print(reg.score(X_test.fillna(0.0),y_test))

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    regr('auditweights_numLikes')
    #learn model
    #model = LogisticRegression(random_state=0, solver='lbfgs',C=0.1).fit(X, y)
