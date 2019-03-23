from cop import *

data = parquet.ParquetDataset(
    # Path to the dataset
    input_path + '/collabTrain/',
    # Dates to read
    filters = [('date','=','2018-02-07')])\
    .read(
    # Columns to read
    columns = col_to_read).to_pandas()
test = parquet.read_table(input_path + '/collabTest').to_pandas()

X_train = preproc_data(data[col_of_int].copy())
X_test = preproc_data(test[col_of_int].copy())
#забив train и test feedback_features
#feedb_train = feat_fr_fb_train(data['feedback'].values)
#X_train, X_test = add_feedback_feat(X_train, feedb_train, X_test)
#del feedb_train
# Construct the label (liked objects)
y_train = data['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0).values
#learn the model
model = xgboost.XGBClassifier(nthread=4,max_depth=10,learning_rate=0.05,n_estimators=150).fit(X_train.values, y_train)
del X_train,y_train
test["predictions"] = -model.predict_proba(X_test.values)[:, 1]
result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
    by=['instanceId_userId', 'predictions'])
submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.to_csv(output_path + "/collabSubmit.csv.gz", header = False, compression='gzip')
