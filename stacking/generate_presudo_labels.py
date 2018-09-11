import numpy as np
import pandas as pd
import gc
from sklearn.metrics import f1_score
from sklearn.linear_model import  SGDClassifier
clf = SGDClassifier(loss='log',n_jobs=-1,max_iter=10,random_state=2018)


strain=np.load('team/train_x.npy')
stest=np.load('team/test_x.npy')
labels=np.load('../data/labels.npy')-1
np.random.seed(1234)
r=(np.random.uniform(0,1,labels.shape[0])*5).astype(np.int32)

sgd_test=np.zeros((102277,19))
sgd_val=np.zeros((102277,19))
for cv_fold in range(5):
    filter_v=(r==cv_fold)
    x_train,y_train=strain[~filter_v],labels[~filter_v]
    x_val,y_val=strain[filter_v],labels[filter_v]
    gc.collect() 
    
    clf.fit(x_train,y_train)
    sgd_val[filter_v,:]=clf.predict_proba(x_val)
    sgd_test+=clf.predict_proba(stest)

sgd_test/=5
print (np.sum(sgd_test,1))
print (f1_score(labels, np.argmax(sgd_val,1), average='macro'))

np.save('stacking.npy',sgd_test)
