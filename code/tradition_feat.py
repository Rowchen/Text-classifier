import sys
column=sys.argv[1]
print (column)
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gc

train = pd.read_csv('../data/train_set.csv',usecols=[column])
test=pd.read_csv('../data/test_set.csv',usecols=[column])

vec = TfidfVectorizer(ngram_range=(1,2),min_df=10, max_df=0.8,use_idf=0,smooth_idf=1,stop_words=['816903','520477'],
                      sublinear_tf=1)

train_term_doc = vec.fit_transform(train[column])
test_term_doc=vec.transform(test[column])
labels=np.load('../data/labels.npy')

from sklearn.linear_model import  SGDClassifier
clf = SGDClassifier(loss='log',n_jobs=-1,max_iter=15,random_state=2018)

gc.collect()
np.random.seed(2018)
r1=(np.random.uniform(0,1,train_term_doc.shape[0])*5).astype(np.int32)

val_tf=np.zeros((102277,19))
test_tf=np.zeros((102277,19))

for cv_fold in range(5):
    
    filter_t=(r1!=cv_fold)
    filter_v=(r1==cv_fold)
    x_train,y_train=train_term_doc[filter_t].copy(),labels[filter_t]
    x_val,y_val=train_term_doc[~filter_t].copy(),labels[~filter_t]

    '''信息增益'''
    smooth=0.00000001
    KL=np.zeros([1,x_train.shape[1]])
    for c in range(1,20):
        filter_c=(y_train==c)
        '''类内散度：该特征出现时该类为C的次数/该特征出现的次数，这个比值越大越好，但是需要做点平滑，因为词语频率很低的词语该项也很小'''
        CD=np.array((np.sum(x_train[filter_c],axis=0)+smooth)/(np.sum(x_train,axis=0)+19*smooth))
        '''计算熵，熵越小越好'''
        KL-=(CD*np.log(CD))
    print (KL.min(),KL.mean(),KL.max())
    KL=KL.max()-KL+0.5
    print (KL.min(),KL.mean(),KL.max())

    gc.collect()
    x_train=x_train.multiply(KL)
    x_val=x_val.multiply(KL)
    x_test=test_term_doc.copy()
    x_test=x_test.multiply(KL)

    clf.fit(x_train,y_train)
    val_tf[filter_v,:] = clf.predict_proba(x_val)
    print (np.mean(np.argmax(val_tf[filter_v,:],1)+1==val_pred))
    
    test_tf+= clf.predict_proba(x_test)
test_tf/=5

np.save('../stacking/tfidf/val_tfidf_%s'%column,val_tf)
np.save('../stacking/tfidf/test_tfidf_%s'%column,test_tf)






    
    
