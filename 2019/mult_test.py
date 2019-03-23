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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
input_path = './'
# Where to store results
output_path = './'
# Read a single day to train model on as Pandas dataframe

col_to_read =['instanceId_userId','feedback','instanceId_objectId', 'audit_pos',
       'audit_timestamp', 'audit_timePassed', 'audit_resourceType',
       'metadata_ownerId', 'metadata_createdAt', 'metadata_authorId',
       'metadata_numPhotos', 'metadata_numPolls', 'metadata_numSymbols',
       'metadata_numTokens', 'metadata_numVideos',
       'userOwnerCounters_USER_FEED_REMOVE',
       'userOwnerCounters_UNKNOWN',
       'userOwnerCounters_CREATE_TOPIC', 'userOwnerCounters_CREATE_IMAGE',
       'userOwnerCounters_CREATE_COMMENT',
       'userOwnerCounters_CREATE_LIKE', 'userOwnerCounters_TEXT',
       'userOwnerCounters_IMAGE', 'userOwnerCounters_VIDEO',
       'membership_statusUpdateDate',
       'membership_joinDate', 'user_create_date', 'user_birth_date',
       'user_gender', 'user_status', 'user_ID_country', 'user_ID_Location',
       'user_change_datime',
       'user_region', 'objectId',
       'auditweights_ageMs', 'auditweights_ctr_gender',
       'auditweights_ctr_high', 'auditweights_ctr_negative',
       'auditweights_dailyRecency', 'auditweights_feedStats',
       'auditweights_friendLikes', 'auditweights_friendLikes_actors',
       'auditweights_isRandom', 'auditweights_likersFeedStats_hyper',
       'auditweights_likersSvd_prelaunch_hyper', 'auditweights_matrix',
       'auditweights_numDislikes', 'auditweights_numLikes',
       'auditweights_numShows', 'auditweights_svd_prelaunch',
       'auditweights_svd_spark', 'auditweights_userAge',
       'auditweights_userOwner_CREATE_LIKE', 'auditweights_userOwner_IMAGE',
       'auditweights_userOwner_TEXT', 'auditweights_x_ActorsRelations']
col_of_int =['instanceId_userId', 'instanceId_objectId', 'audit_pos',
       'audit_timestamp', 'audit_timePassed', 'audit_resourceType',
       'metadata_ownerId', 'metadata_createdAt', 'metadata_authorId',
       'metadata_numPhotos', 'metadata_numPolls', 'metadata_numSymbols',
       'metadata_numTokens', 'metadata_numVideos',
       'userOwnerCounters_USER_FEED_REMOVE',
       'userOwnerCounters_UNKNOWN',
       'userOwnerCounters_CREATE_TOPIC', 'userOwnerCounters_CREATE_IMAGE',
       'userOwnerCounters_CREATE_COMMENT',
       'userOwnerCounters_CREATE_LIKE', 'userOwnerCounters_TEXT',
       'userOwnerCounters_IMAGE', 'userOwnerCounters_VIDEO',
       'membership_statusUpdateDate',
       'membership_joinDate', 'user_create_date', 'user_birth_date',
       'user_gender', 'user_status', 'user_ID_country', 'user_ID_Location',
       'user_change_datime',
       'user_region', 'objectId',
       'auditweights_ageMs', 'auditweights_ctr_gender',
       'auditweights_ctr_high', 'auditweights_ctr_negative',
       'auditweights_dailyRecency', 'auditweights_feedStats',
       'auditweights_friendLikes', 'auditweights_friendLikes_actors',
       'auditweights_isRandom', 'auditweights_likersFeedStats_hyper',
       'auditweights_likersSvd_prelaunch_hyper', 'auditweights_matrix',
       'auditweights_numDislikes', 'auditweights_numLikes',
       'auditweights_numShows', 'auditweights_svd_prelaunch',
       'auditweights_svd_spark', 'auditweights_userAge',
       'auditweights_userOwner_CREATE_LIKE', 'auditweights_userOwner_IMAGE',
       'auditweights_userOwner_TEXT', 'auditweights_x_ActorsRelations']

def partitioned_auc(data):
    return data.groupby("instanceId_userId")\
        .apply(lambda y: auc(y.label.values, y.score.values))\
        .dropna()
def auc(labels, scores):
    # This is important! AUC can be computed only when both positive and negative examples are
    # available
    if len(labels) > sum(labels) > 0:
        return roc_auc_score(labels, scores)
    return float('NaN')
#забив nan регрессией
def regr_data(df_train:pd.DataFrame,df_pred_tr:pd.DataFrame,df_pred_test:pd.DataFrame, col:str):
    if col in df_train.columns:
        y = df_train[df_train[col].isna()==False][col].values
        X = df_train[df_train[col].isna()==False]
        X = X.fillna(0.0).drop([col],axis=1).values
        print('st_learn '+col)
        #reg = xgboost.XGBRegressor(nthread=4).fit(X,y)
        reg = xgboost.XGBRegressor().fit(X,y)
        del X, y
        print('st_learn_end '+col)
        df_pred_tr.loc[df_pred_tr[col].isna()==True,col] = \
            reg.predict(df_pred_tr.loc[df_pred_tr[col].isna()==True,df_pred_tr.columns].drop([col],inplace=False,axis=1).fillna(0.0).values)
        df_pred_test.loc[df_pred_test[col].isna()==True,col] = \
            reg.predict(df_pred_test.loc[df_pred_test[col].isna()==True,df_pred_test.columns].drop([col],inplace=False,axis=1).fillna(0.0).values)

    return [df_pred_tr,df_pred_test]
#features from feedback train
def feat_fr_fb_train(feedb:np.ndarray, thr = 0):
    feedb = [' '.join(x.tolist()) for x in feedb]
    vectorizer = CountVectorizer()
    feedb = vectorizer.fit_transform(feedb)
    feedb = feedb.toarray()
    feedb = pd.DataFrame(feedb,columns=vectorizer.get_feature_names())
    feedb.drop('liked',axis=1,inplace=True)
    for name in feedb.columns:
        if feedb[name].sum()<thr :
            feedb.drop(name, axis=1, inplace=True)
    return  feedb
#предобработка данных
def preproc_data(df_train:pd.DataFrame,df_test:pd.DataFrame):
    '''
    if 'audit_timestamp' in df.columns:
        df['audit_timestamp'] = (df['audit_timestamp'] - np.min(df['audit_timestamp'].values))\
    /(np.max(df['audit_timestamp'].values)-np.min(df['audit_timestamp'].values))

    #if 'auditweights_ctr_high' in df.columns:
        #df.loc[df['auditweights_ctr_high']>0.6] = 0.6

    if 'auditweights_ctr_gender' in df.columns:
        df['auditweights_ctr_gender'].fillna(np.median(df['auditweights_ctr_gender'].fillna(0.0)),inplace=True)
        df.loc[df['auditweights_ctr_gender']>0.1] = 0.1

    if 'auditweights_friendLikes' in df.columns:
        df.loc[df['auditweights_friendLikes']>1] = 1

    #if 'auditweights_dailyRecency' in df.columns:
        #df.loc[df['auditweights_dailyRecency']<0.7] = 0.7

    #if 'auditweights_matrix' in df.columns:
        #df['auditweights_matrix'].fillna(np.median(df['auditweights_matrix'].fillna(0.0)),inplace=True)
        #df.loc[df['auditweights_matrix']>0.1] = 0.1

    if 'auditweights_numDislikes' in df.columns:
        #df.loc[df['auditweights_numDislikes']>0.1] = 0.1
        df['auditweights_numDislikes'].fillna(np.median(df['auditweights_numDislikes'].fillna(0.0)),inplace=True)

    if 'auditweights_userOwner_CREATE_LIKE' in df.columns:
        df['auditweights_userOwner_CREATE_LIKE'].fillna(np.median(df['auditweights_userOwner_CREATE_LIKE'].fillna(0.0)),inplace=True)

    if 'auditweights_userOwner_TEXT' in df.columns:
        df['auditweights_userOwner_TEXT'].fillna(np.median(df['auditweights_userOwner_TEXT'].fillna(0.0)),inplace=True)
    '''
    #regr_data(df,'auditweights_userOwner_CREATE_LIKE')
    #regr_data(df,'auditweights_ctr_negative')
    #df['auditweights_numShows'].fillna(np.median(df['auditweights_numShows'].fillna(0.0)),inplace=True)
    #regr_data(df,'auditweights_numDislikes')


    data = parquet.read_table(input_path + '/collabTrain/date=2018-03-19').to_pandas()
    data = data[col_of_int]

    df_train,df_test = regr_data(data.fillna(0.0),df_train,df_test,'auditweights_numLikes')
    df_train,df_test = regr_data(data.fillna(0.0),df_train,df_test,'auditweights_svd_spark')
    df_train,df_test = regr_data(data.fillna(0.0),df_train,df_test,'auditweights_svd_prelaunch')
    del data


    #regr_data(df,'auditweights_svd_prelaunch')
    #regr_data(df,'auditweights_svd_spark')
    #regr_data(df,'userOwnerCounters_CREATE_LIKE')
    return [df_train,df_test]

def add_feedback_feat(train:pd.DataFrame,feedb_train:pd.DataFrame,test:pd.DataFrame):
    test_col = test.columns.tolist()
    for name in ['commented', 'complaint', 'disliked', 'reshared', 'unliked', 'viewed']:
        clf = xgboost.XGBClassifier(nthread=4,max_depth=5,learning_rate=0.1,n_estimators=50,min_child_weight=3).fit(train.values, feedb_train[name])
        test[name] = clf.predict(test[test_col].values)
        print('classif',name,sep=': ')
    train = pd.concat([train,feedb_train[['commented', 'complaint', 'disliked', 'reshared', 'unliked', 'viewed']]],axis=1)
    clf = xgboost.XGBClassifier(nthread=4,max_depth=5,learning_rate=0.1,n_estimators=50,min_child_weight=3).fit(train.values, feedb_train['clicked'])
    test['clicked'] = clf.predict(test.values)
    train = pd.concat([train,feedb_train['clicked']],axis=1)
    print('classif','clicked',sep=': ')
    #print(test['clicked'])
    #clf = xgboost.XGBClassifier(nthread=4,max_depth=5,learning_rate=0.1,n_estimators=50,min_child_weight=3).fit(train.values, feedb_train['ignored'])
    #test['ignored'] = clf.predict(test.values)
    #train = pd.concat([train,feedb_train['ignored']],axis=1)
    return [train,test]

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    train_dates = [[('date', '=', '2018-02-' + x)] for x in ['20']]
    data = parquet.ParquetDataset(
        # Path to the dataset
        input_path + '/collabTrain/',
        # Dates to read
        filters = train_dates)\
        .read(
        # Columns to read
        columns = col_to_read).to_pandas()

    test_dates = [[('date', '=', '2018-02-' + x)] for x in ['21']]
    test = parquet.ParquetDataset(input_path + '/collabTrain/', filters = test_dates)\
        .read(columns = col_to_read).to_pandas()

    X_train,X_test = preproc_data(data[col_of_int].copy(),test[col_of_int].copy())

    #забив train и test feedback_features
    #feedb_train = feat_fr_fb_train(data['feedback'].values)
    #X_train, X_test = add_feedback_feat(X_train, feedb_train, X_test)
    #del feedb_train
    # Construct the label (liked objects)
    y_train = data['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0).values
    #learn the model
    model = xgboost.XGBClassifier(nthread=4,max_depth=5,learning_rate=0.1,n_estimators=50,min_child_weight=3).fit(X_train,y_train)
    #model = ExtraTreesClassifier()
    #model.fit(X_train.fillna(0.0), y_train)
    #l = list(zip(model.feature_importances_,col_of_int))
    #for i in l:
        #print(i)

    #rfe = RFE(model, 20)
    #fit = rfe.fit(X_train.fillna(0.0), y_train)
    #print(fit.support_)
    #xgboost.plot_importance(model)
    #plt.show()
    del X_train,y_train


    test["score"] = model.predict_proba(X_test)[:, 1]
    # Extract labels and project
    test["label"] = test['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0)

    test = test[["instanceId_userId", "score", "label"]]

    max_user = max(test.instanceId_userId)
    batch_size = 1000000
    batches = [test[test.instanceId_userId.between(x, x + batch_size)] for x in range(0,max_user,batch_size)]
    with Pool(int(cpu_count() / 2)) as p:
        ret_list = p.map(
            partitioned_auc,
            batches)

    print(pd.concat(ret_list).mean())
