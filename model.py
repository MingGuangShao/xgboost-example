import os
import csv
import xgboost as xgb
import numpy as np
import cPickle as pickle
from sklearn import cross_validation, metrics
from xgboost.sklearn import XGBClassifier

path = "./pickle/"
result_path = "./result/" 
files = os.listdir(path)
#read data
if not os.path.isdir(files[0]):
    print "load file, ",files[0]
    with open(path + files[0], 'rb') as f:
        data = pickle.load(f)

for fname in files[1:]:
    if not os.path.isdir(fname):
        print "load file, ",fname
        with open(path + fname, 'rb') as f:
            data = np.concatenate((data,pickle.load(f)),axis = 0)

print "shape of data",data.shape
#start train
train_data = data[:,:-1]
train_label = data[:,-1]
print "shape of train_data",train_data.shape
print "shape of train_label",train_label.shape

xg_train = xgb.DMatrix(train_data, label = train_label)

param = {'eta':0.05, 'silent':1, 'min_child_weight':0, 'gamma':0, 'subsample':0.8, 'lambda':0.9, 'colsample_bytree':0.9, 'objective':'reg:linear'}
param['eval_metric'] = ['rmse', 'map']
watchlist = [ (xg_train,'train') ]

#train_model
num_round = 1000
bst = xgb.train(param, xg_train, num_round, watchlist, verbose_eval=10 )

#calculate feature importance
with open(result_path + 'feature_importance.csv', 'wb') as fout:
  scores = bst.get_fscore()
  for line in scores:
    fout.write(line+','+str(scores[line])+'\n')

# cross validation
print ('running cross validation, with preprocessing function')

# define the preprocessing function
# we can use this to do weight rescale, etc.
# as a example, we try to set scale_pos_weight
num_round = 0
re = xgb.cv(param, xg_train, num_round, nfold=5, verbose_eval=1, metrics={'rmse', 'map'}, seed = 0)

print(re)
re.to_csv(result_path + 'cv.csv', encoding='utf-8', index=True)

# save out model
#bst.save_model('eta0.1_subsample0.5_ams0.15.model')
# make prediction
#result = bst.predict(xg_test)

