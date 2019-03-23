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
from sklearn.decomposition import PCA

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

#features from feedback train
def feat_fr_fb_train(feedb:np.ndarray, thr = 1000000):
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

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    input_path = './'
    # Where to store results
    output_path = './'
    # Read a single day to train model on as Pandas dataframe
    col_to_read = [
        'instanceId_userId',
        'feedback',
        'instanceId_objectType',
        'auditweights_ageMs',
        'auditweights_ctr_gender',
        'auditweights_ctr_high',
        'auditweights_ctr_negative',
        'auditweights_dailyRecency',
        'auditweights_feedStats',
        'auditweights_friendLikes',
        'auditweights_friendLikes_actors',
        'auditweights_isRandom',
        'auditweights_likersFeedStats_hyper',
        'auditweights_likersSvd_prelaunch_hyper',
        'auditweights_matrix',
        'auditweights_numDislikes',
        'auditweights_numLikes',
        'auditweights_numShows',
        'auditweights_svd_prelaunch',
        'auditweights_userAge',
        'auditweights_userOwner_CREATE_LIKE',
        'auditweights_userOwner_TEXT']
    col_of_int = [
            'instanceId_objectType',
            'auditweights_ageMs',
            'auditweights_ctr_gender',
            'auditweights_ctr_high',
            'auditweights_ctr_negative',
            'auditweights_dailyRecency',
            'auditweights_feedStats',
            'auditweights_friendLikes',
            'auditweights_friendLikes_actors',
            'auditweights_isRandom',
            'auditweights_likersFeedStats_hyper',
            'auditweights_likersSvd_prelaunch_hyper',
            'auditweights_matrix',
            'auditweights_numDislikes',
            'auditweights_numLikes',
            'auditweights_numShows',
            'auditweights_svd_prelaunch',
            'auditweights_userAge',
            'auditweights_userOwner_CREATE_LIKE',
            'auditweights_userOwner_TEXT']
    data = parquet.ParquetDataset(
        # Path to the dataset
        input_path + '/collabTrain/',
        # Dates to read
        filters = [('date','=','2018-02-07')])\
        .read(
        # Columns to read
        columns = col_to_read).to_pandas()

    # Construct the label (liked objects)
    y = data['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0).values

    df = data[col_of_int]
    dum = pd.get_dummies(df['instanceId_objectType'].values)
    df.drop(['instanceId_objectType'],axis=1,inplace=True)
    df = pd.concat([df,dum],axis=1)
    X_0 = df.fillna(0.0).values
    # Extract the most interesting features
    #X = df.fillna(0.0).values
    #learn model
    #model = LogisticRegression(random_state=0, solver='lbfgs',C=0.1).fit(X, y)
    for i in range(4,df.shape[0],1):
        X = X_0
        pca = PCA(n_components=i)
        X = pca.fit_transform(X)
        model = xgboost.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,min_child_weight=3).fit(X, y)
    #model =  CatBoostClassifier(learning_rate=0.1,depth=5).fit(X, y)
    # Pick one week for the test period
        test_dates = [[('date', '=', '2018-02-' + x)] for x in ['08','09','10','11','12','13','14']]

    # Read the test data for those days, only required columns
        test = parquet.ParquetDataset(input_path + '/collabTrain/', filters = test_dates)\
            .read(columns = col_to_read).to_pandas()



        df = test[col_of_int]
        dum = pd.get_dummies(df['instanceId_objectType'].values)
        df.drop(['instanceId_objectType'],axis=1,inplace=True)
        df = pd.concat([df,dum],axis=1)
        X_test = df.fillna(0.0)
        X_test = pca.transform(X_test)
    #test['ignore'] = np.random.choice([0, 1], size=(test.shape[0],), p=[1./3,2./3])
            # Start processing of the batches in several threads
        # Compute inverted predictions (to sort by later)
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

        print(i,pd.concat(ret_list).mean(),sep = ': ')
