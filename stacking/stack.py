import numpy as np
import pandas as pd
import gc
from sklearn.metrics import f1_score

team_train=np.load('team/train_x.npy')
team_test=np.load('team/test_x.npy')
labels=np.load('../data/labels.npy')-1


'''start merge data'''
np.random.seed(2018)
r=(np.random.uniform(0,1,labels.shape[0])*5).astype(np.int32)
rnn1_val=np.zeros((labels.shape[0],19))
rnn2_val=np.zeros((labels.shape[0],19))
rnn3_val=np.zeros((labels.shape[0],19))
rnn4_val=np.zeros((labels.shape[0],19))
rnn5_val=np.zeros((labels.shape[0],19))

cnn_val=np.zeros((labels.shape[0],19))
rcnn_val=np.zeros((labels.shape[0],19))

fast1_val=np.zeros((labels.shape[0],19))
fast2_val=np.zeros((labels.shape[0],19))
fast3_val=np.zeros((labels.shape[0],19))

tw_val=np.load('tfidf/val_tfidf_word_seg.npy')
tc_val=np.load('tfidf/val_tfidf_article.npy')

for cv in range(1,6):
    filter_v=(r==cv-1)
    rnn1_val[filter_v]=np.load('rnn/val_rnn1_%d.npy'%cv)
    rnn2_val[filter_v]=np.load('rnn/val_rnn2_%d.npy'%cv)
    rnn3_val[filter_v]=np.load('rnn/val_rnn3_%d.npy'%cv)
    rnn4_val[filter_v]=np.load('rnn/val_rnn4_%d.npy'%cv)
    rnn5_val[filter_v]=np.load('rnn/val_rnn5_%d.npy'%cv)

    fast1_val[filter_v]=np.load('fast/val_fast1_%d.npy'%cv)
    fast2_val[filter_v]=np.load('fast/val_fast2_%d.npy'%cv)
    fast3_val[filter_v]=np.load('fast/val_fast3_%d.npy'%cv)
    
    cnn_val[filter_v]=np.load('cnn/val_cnn_%d.npy'%cv)
    rcnn_val[filter_v]=np.load('rcnn/val_rcnn_%d.npy'%cv)

    
print ('rnn1_val',f1_score(labels, np.argmax(rnn1_val,1), average='macro'))
print ('rnn2_val',f1_score(labels, np.argmax(rnn2_val,1), average='macro'))
print ('rnn3_val',f1_score(labels, np.argmax(rnn3_val,1), average='macro'))
print ('rnn4_val',f1_score(labels, np.argmax(rnn4_val,1), average='macro'))
print ('rnn5_val',f1_score(labels, np.argmax(rnn5_val,1), average='macro'))
print ('fast1_val',f1_score(labels, np.argmax(fast1_val,1), average='macro'))
print ('fast2_val',f1_score(labels, np.argmax(fast2_val,1), average='macro'))
print ('fast3_val',f1_score(labels, np.argmax(fast3_val,1), average='macro'))
print ('cnn_val',f1_score(labels, np.argmax(cnn_val,1), average='macro'))
print ('rcnn_val',f1_score(labels, np.argmax(rcnn_val,1), average='macro'))
print ('tw_val',f1_score(labels, np.argmax(tw_val,1), average='macro'))
print ('tc_val',f1_score(labels, np.argmax(tc_val,1), average='macro'))



'''start merge data'''
np.random.seed(2018)
r=(np.random.uniform(0,1,labels.shape[0])*5).astype(np.int32)
rnn1_test=np.zeros((labels.shape[0],19))
rnn2_test=np.zeros((labels.shape[0],19))
rnn3_test=np.zeros((labels.shape[0],19))
rnn4_test=np.zeros((labels.shape[0],19))
rnn5_test=np.zeros((labels.shape[0],19))

cnn_test=np.zeros((labels.shape[0],19))
rcnn_test=np.zeros((labels.shape[0],19))

fast1_test=np.zeros((labels.shape[0],19))
fast2_test=np.zeros((labels.shape[0],19))
fast3_test=np.zeros((labels.shape[0],19))

tw_test=np.load('tfidf/test_tfidf_word_seg.npy')
tc_test=np.load('tfidf/test_tfidf_article.npy')

for cv in range(1,6):
    filter_v=(r==cv-1)
    rnn1_test+=(np.load('rnn/test_rnn1_%d.npy'%cv)/5)
    rnn2_test+=(np.load('rnn/test_rnn2_%d.npy'%cv)/5)
    rnn3_test+=(np.load('rnn/test_rnn3_%d.npy'%cv)/5)
    rnn4_test+=(np.load('rnn/test_rnn4_%d.npy'%cv)/5)
    rnn5_test+=(np.load('rnn/test_rnn5_%d.npy'%cv)/5)
    fast1_test+=(np.load('fast/test_fast1_%d.npy'%cv)/5)
    fast2_test+=(np.load('fast/test_fast2_%d.npy'%cv)/5)
    fast3_test+=(np.load('fast/test_fast3_%d.npy'%cv)/5)
    cnn_test+=(np.load('cnn/test_cnn_%d.npy'%cv)/5)
    rcnn_test+=(np.load('rcnn/test_rcnn_%d.npy'%cv)/5)
    
strain=np.concatenate([team_train,rnn1_val,rnn2_val,rnn3_val,rnn4_val,rnn5_val,cnn_val,rcnn_val,
                     fast1_val,fast2_val,fast3_val,tw_val,tc_val],1)
stest=np.concatenate([team_test,rnn1_test,rnn2_test,rnn3_test,rnn4_test,rnn5_test,cnn_test,rcnn_test,
                     fast1_test,fast2_test,fast3_test,tw_test,tc_test],1)

np.random.seed(1234)
r=(np.random.uniform(0,1,labels.shape[0])*5).astype(np.int32)


from sklearn.linear_model import  SGDClassifier
clf = SGDClassifier(loss='log',n_jobs=-1,max_iter=10,random_state=2018)

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
print (f1_score(labels, np.argmax(sgd_val,1), average='macro') )


import lightgbm as lgb
clf = lgb.LGBMClassifier(objective='multiclass',
                                max_depth=5,
                                num_leaves=16,
                                learning_rate=0.06,
                                n_estimators=2000,
                                colsample_bytree=0.4,#0.3
                                subsample = 0.75,#0.75
                                n_jobs=-1,
                                lambda_l2=10,
                                seed=2018
                                )

lgb_test=np.zeros((102277,19))
lgb_val=np.zeros((102277,19))
for cv_fold in range(5):
    filter_v=(r==cv_fold)
    x_train,y_train=strain[~filter_v],labels[~filter_v]
    x_val,y_val=strain[filter_v],labels[filter_v]
    gc.collect() 
    
    clf.fit(x_train,y_train,eval_set=[(x_val,y_val)],
                    eval_metric=['multi_logloss'],
                    early_stopping_rounds=40,verbose=40)
    lgb_val[filter_v,:]=clf.predict_proba(x_val)
    lgb_test+=clf.predict_proba(stest)

lgb_test/=5
print (np.sum(lgb_test,1))
print (f1_score(labels, np.argmax(lgb_val,1), average='macro') )


from sklearn.linear_model import  SGDClassifier
clf = SGDClassifier(loss='hinge',n_jobs=-1,max_iter=15,random_state=2018)

hinge_test=np.zeros((102277,19))
hinge_val=np.zeros((102277,19))
for cv_fold in range(5):
    filter_v=(r==cv_fold)
    x_train,y_train=strain[~filter_v],labels[~filter_v]
    x_val,y_val=strain[filter_v],labels[filter_v]
    gc.collect() 
    
    clf.fit(x_train,y_train)
    a=clf.decision_function(x_val)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    hinge_val[filter_v,:]=d
    
    a=clf.decision_function(stest)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    hinge_test+=d

hinge_test/=5
print (np.sum(hinge_test,1))
print (f1_score(labels, np.argmax(hinge_val,1), average='macro') )



from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=2018)

svc_test=np.zeros((102277,19))
svc_val=np.zeros((102277,19))
for cv_fold in range(5):
    filter_v=(r==cv_fold)
    x_train,y_train=strain[~filter_v],labels[~filter_v]
    x_val,y_val=strain[filter_v],labels[filter_v]
    gc.collect() 
    
    clf.fit(x_train,y_train)

    a=clf.decision_function(x_val)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    svc_val[filter_v,:]=d
    
    a=clf.decision_function(stest)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    svc_test+=d

svc_test/=5
print (np.sum(svc_test,1))
print (f1_score(labels, np.argmax(svc_val,1), average='macro') )


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=4, dual=True,random_state=2018)

lr_test=np.zeros((102277,19))
lr_val=np.zeros((102277,19))
for cv_fold in range(5):
    filter_v=(r==cv_fold)
    x_train,y_train=strain[~filter_v],labels[~filter_v]
    x_val,y_val=strain[filter_v],labels[filter_v]
    gc.collect() 
    
    clf.fit(x_train,y_train)
    lr_val[filter_v,:]=clf.predict_proba(x_val)
    lr_test+=clf.predict_proba(stest)

lr_test/=5
print (np.sum(lr_test,1))
print (f1_score(labels, np.argmax(lr_val,1), average='macro') )

from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=2018)

np.random.seed(1000)
r=(np.random.uniform(0,1,labels.shape[0])*5).astype(np.int32)
stack_test=np.zeros((102277,19))
stack_val=np.zeros((102277,19))
for cv_fold in range(5):
    filter_v=(r==cv_fold)
    x_train,y_train=st_train[~filter_v],labels[~filter_v]
    x_val,y_val=st_train[filter_v],labels[filter_v]
    gc.collect() 
    
    clf.fit(x_train,y_train)

    a=clf.decision_function(x_val)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    stack_val[filter_v,:]=d
    
    a=clf.decision_function(st_test)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    stack_test+=d

stack_test/=5
print (np.sum(stack_test,1))
print (f1_score(labels, np.argmax(stack_val,1), average='macro') )


'''stack_level2'''

st_train=np.concatenate((svc_val,lr_val,lgb_val,sgd_val,hinge_val),1)
st_test=np.concatenate((svc_test,lr_test,lgb_test,sgd_test,hinge_test),1)
print (st_train.shape,st_test.shape)

np.random.seed(1000)
r=(np.random.uniform(0,1,labels.shape[0])*5).astype(np.int32)
stack_test=np.zeros((102277,19))
stack_val=np.zeros((102277,19))
for cv_fold in range(5):
    filter_v=(r==cv_fold)
    x_train,y_train=st_train[~filter_v],labels[~filter_v]
    x_val,y_val=st_train[filter_v],labels[filter_v]
    gc.collect() 
    
    clf.fit(x_train,y_train)

    a=clf.decision_function(x_val)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    stack_val[filter_v,:]=d
    
    a=clf.decision_function(st_test)
    b=a-np.max(a,1,keepdims=True)
    c=np.exp(b)
    d=c/np.sum(c,1,keepdims=True)
    stack_test+=d

stack_test/=5
print (np.sum(stack_test,1))
print (f1_score(labels, np.argmax(stack_val,1), average='macro') )


stack_result=np.argmax(stack_test,1)+1
print (len(stack_result))
submit=pd.DataFrame({'id':list(range(len(stack_result))),'class':stack_result})
submit.to_csv('../submit/submit.csv',index=None,sep=',')
