from mult_test import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    train = parquet.read_table(input_path + '/collabTrain/date=2018-03-20').to_pandas()
    test = parquet.read_table(input_path + '/collabTrain/date=2018-03-21').to_pandas()
    df_1 = feat_fr_fb_train(train['feedback'])
    df_2 = feat_fr_fb_train(test['feedback'])
    train = preproc_data(train[col_of_int])
    test = preproc_data(test[col_of_int])
    train = pd.concat([train,df_1[['commented', 'complaint', 'disliked', 'reshared', 'unliked', 'viewed']]],axis=1)
    test = pd.concat([test,df_2[['commented', 'complaint', 'disliked', 'reshared', 'unliked', 'viewed']]],axis=1)
    for name in ['clicked']:
        clf = xgboost.XGBClassifier(nthread=4,max_depth=5,learning_rate=0.1,n_estimators=50,min_child_weight=3).fit(train, df_1[name])
        #bst = xgboost.Booster()
        #bst.set_score = clf.score(test,df_2[name])
        xgboost.plot_importance(clf)
        plt.show()
        print(name,clf.score(test,df_2[name]),sep=': ')
