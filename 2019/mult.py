from mult_test import *
def feat_fr_fb_test(feedb:np.ndarray,feedb_col:list):
    feedb = [' '.join(x.tolist()) for x in feedb]
    vectorizer = CountVectorizer()
    feedb = vectorizer.fit_transform(feedb)
    feedb = feedb.toarray()
    feedb = pd.DataFrame(feedb,columns=vectorizer.get_feature_names())
    feedb = feedb[feedb_col]
    return feedb


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    data = parquet.ParquetDataset(
        # Path to the dataset
        input_path + '/collabTrain/',
        # Dates to read
        filters = [('date','=','2018-03-07')])\
        .read(
        # Columns to read
        columns = col_to_read).to_pandas()

    test_dates = [[('date', '=', '2018-03-' + x)] for x in ['08','09','10','11','12','13','14']]
    test = parquet.ParquetDataset(input_path + '/collabTrain/', filters = test_dates)\
        .read(columns = col_to_read).to_pandas()

    X_train = preproc_data(data[col_of_int])
    X_test = preproc_data(test[col_of_int])

    #забив train и test feedback_features
    feedb_train = feat_fr_fb_train(data['feedback'].values)
    X_train, X_test = add_feedback_feat(X_train, feedb_train, X_test)
    del feedb_train
    # Construct the label (liked objects)
    y_train = data['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0).values
    #learn the model
    model = xgboost.XGBClassifier(nthread=4,max_depth=5,learning_rate=0.1,n_estimators=50,min_child_weight=3).fit(X_train.values, y_train)

    del X_train,y_train

    X_test = X_test.drop(['commented', 'complaint', 'disliked', 'reshared', 'unliked', 'viewed', 'clicked', 'ignored'],axis=1)
    df_2 = feat_fr_fb_test(test['feedback'].values,['commented', 'complaint', 'disliked', 'reshared', 'unliked', 'viewed', 'clicked', 'ignored'])

    X_test = pd.concat([X_test,df_2],axis=1)

    test["score"] = model.predict_proba(X_test.values)[:, 1]

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
