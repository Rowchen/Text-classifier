import pandas as pd, numpy as np
from tqdm import tqdm

column='word_seg'
labels=pd.read_csv('../data/train_set.csv',usecols=['class']).values
labels=labels.reshape(-1)
np.save('../data/labels.npy',labels)
train = pd.read_csv('../data/train_set.csv',usecols=[column])
test=pd.read_csv('../data/test_set.csv',usecols=[column])
alldoc=np.concatenate((train[column].values,test[column].values),axis=0)

import collections
def build_vocab(data):
    ls=collections.Counter()
    for row in tqdm(range(data.shape[0])):
        ls.update(collections.Counter(data[row].split()))
    return ls
import operator
word=build_vocab(alldoc)
temp = sorted(word.items(),key=operator.itemgetter(1),reverse=True)

word=dict(filter(lambda x: (x[1]>1)&(x[1]<4000000),temp))
word2idx={}
for i,k in enumerate(word):
    word2idx[k]=i
idx2word=list(word)
print (len(idx2word))

def build_word(data,word2idx,maxlen):
    ls=data[column].values
    embed=np.ones((ls.shape[0],maxlen),dtype=np.int32)*679249
    for row in tqdm(range(ls.shape[0])):
        s=ls[row].split()
        cnt=0
        for w in s:
            if w in word2idx:
                embed[row,cnt]=word2idx[w]
                cnt+=1
            if cnt>=maxlen:
                break
    return embed

train_embed=build_word(train,word2idx,maxlen=1000)
test_embed=build_word(test,word2idx,maxlen=1000)
import gc
gc.collect()
np.save('../data/train_embed.npy',train_embed)
np.save('../data/test_embed.npy',test_embed)

print ('using glove to train')
alldoc=pd.concat([train,test])
alldoc.to_csv('alldoc.txt',header=None,index=None)
import subprocess
subprocess.call('./glove.sh',shell=True)

with open('glove/vectors.txt', 'r+') as f:
    content = f.read()        
    f.seek(0, 0)
    f.write('679242 100\n'+content)
    
from gensim.models import Word2Vec
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('glove/vectors.txt', binary=False)


word_vec=np.zeros([679250,100],dtype=np.float32)
cnt=0
for i in range(679242):
    try:
        word_vec[i]=model.wv.word_vec(idx2word[i])
    except:
        print (idx2word[i],word[idx2word[i]])
        word_vec[i]=np.random.rand()
print (cnt)
np.save('../data/glove.npy',word_vec)


alldoc=np.concatenate((train[column].values,test[column].values),axis=0)
print ('now train word2vec')
import gensim
TaggededDocument = gensim.models.doc2vec.TaggedDocument
class sentences_generator():
    def __init__(self, doc):
        self.doc = doc
    def __iter__(self):
        for line in self.doc:
            sentence = line.split()
            yield sentence

from gensim.models import word2vec
sents=sentences_generator(alldoc)
print ('start training,need 2hours or more')
model = word2vec.Word2Vec(sents, sg=1, size=100, window=5, min_count=2, hs=1, workers=8,iter=20)
word_vec=np.zeros([679250,100],dtype=np.float32)
for i in range(679242):
    word_vec[i]=model.wv.word_vec(idx2word[i])
np.save('../data/word_vec.npy',word_vec)






