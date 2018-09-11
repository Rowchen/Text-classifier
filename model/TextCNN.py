import tensorflow as tf
import numpy as np
import pandas as pd
vector1=np.load('../data/word_vec.npy')
word_embed_vector=vector1
print (word_embed_vector.shape)

class param(object):
    num_classes=19
    sequence_length=1000
    embed_size=100
    vocab_size=679250
    batch_size=128
    epoch=1
    
class cnn_param(param):
    filter_sizes=[1,2,3,4]
    filter_num=[256,256,256,256]
    drop_keep_prob=0.5
    l2_lambda=1e-3
    hiddim=100

arg=cnn_param()


class Basic_model:
    def __init__(self,num_classes,sequence_length,vocab_size,embed_size):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.global_steps=tf.Variable(0, trainable=False)
        self.embed=tf.Variable(word_embed_vector,name='embeding_vector')
        
        #placeholder
        self.x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")  # X
        self.y = tf.placeholder(tf.float32,[None,19], name="labels")
        self.keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.training=tf.placeholder(tf.bool,name="training")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        self.lr_embed = tf.placeholder(tf.float32, name="embed_learning_rate")
        self.lamda=tf.placeholder(tf.float32, name="l2_regular")
        
    def weight_init(self,shape,name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            weight=tf.get_variable('kernel',shape,initializer=tf.contrib.layers.xavier_initializer())
        return  weight
    
    def bias_init(self,shape,name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            bias=tf.Variable(tf.zeros(shape)+0.1,tf.float32,name='bias')
        return  bias
    
class TextCNN(Basic_model):
    def __init__(self,arg):
        super(TextCNN, self).__init__(arg.num_classes,arg.sequence_length,arg.vocab_size,arg.embed_size)
        
        self.filter_sizes=arg.filter_sizes  # list,eg[1,2,3,4,5]
        self.filter_num=arg.filter_num #list,eg[32,32,64]
        self.hiddim=arg.hiddim
        
        #conv filter paramaeter
        self.filters=[self.weight_init([self.filter_sizes[i],self.embed_size,1,self.filter_num[i]],name='filter%d'%i)
                       for i in range(len(self.filter_sizes))]
        self.bias=[self.bias_init([self.filter_num[i]],name='filter%d'%i) for i in range(len(self.filter_sizes))]
        
        self.logit=self.forward()
        self.proba=tf.nn.softmax(self.logit,axis=1)
          
        self.losses=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logit))
        self.spareloss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.y,1), logits=self.logit))
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name and 'dense'in v.name]) * self.lamda
        self.loss_add_reg=self.losses+self.l2_losses
        [print(v) for v in tf.trainable_variables() if 'kernel' in v.name and 'dense'in v.name]
        
        self.acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logit,1),tf.argmax(self.y,1)),tf.float32))
        
        var1 = [v for v in tf.trainable_variables() if 'embeding_vector' in v.name]
        var2 = [v for v in tf.trainable_variables() if 'embeding_vector' not in v.name]
        print ('pretrained,fine-tuning',var1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step1=tf.train.AdamOptimizer(self.lr_embed).minimize(self.loss_add_reg,var_list=var1)
            self.train_step2=tf.train.AdamOptimizer(self.lr).minimize(self.loss_add_reg,global_step=self.global_steps,var_list=var2)
            self.train_op = tf.group(self.train_step1, self.train_step2)
        
    def conv(self,x,W,b):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')+b
    
    def forward(self):
        
        s = tf.expand_dims(tf.nn.embedding_lookup(self.embed,self.x),-1)#[None,sentence_length,embed_size,1]
        print ('s',s.shape)
        
        o1=[tf.nn.relu((self.conv(s,self.filters[i],self.bias[i]))) for i in range(len(self.filters))]
        [print ('o1',o1[i].shape) for i in range(len(self.filters))]
    
    
        o2=tf.concat([tf.nn.top_k(tf.transpose(o1[i],[0,2,3,1]),k=3,sorted=False)[0] for i in range(len(self.filters))],-2)
        print ('o2',o2.shape)
        o4=tf.reshape(o2,[-1,o2.shape[2]*o2.shape[3]])

        print (o4.shape)
        
        o4drop=tf.nn.dropout(o4,self.keep_prob)
        print('o4drop',o4drop.shape)
        
        o5=tf.layers.dense(o4drop,self.hiddim,activation=None,use_bias=True,
                           kernel_initializer=tf.contrib.layers.xavier_initializer()
                           ,kernel_regularizer=None)
        o5bn=tf.nn.relu(tf.layers.batch_normalization(o5,training=self.training))
        o5drop=tf.nn.dropout(o5bn,self.keep_prob)
        print('o5drop',o5drop.shape)
        
        o6=tf.layers.dense(o5drop,self.num_classes,activation=None,use_bias=True,
                           kernel_initializer=tf.contrib.layers.xavier_initializer()
                           ,kernel_regularizer=None)
        print ('o6',o6.shape)
        return o6   

tf.reset_default_graph()
ss=TextCNN(arg)

import sys
use_test=[]
cv_fold=int(sys.argv[1])
print (cv_fold)
if cv_fold==1:
    use_test=list(range(50001))
if cv_fold==2:
    use_test=list(range(50000,102277))
if cv_fold==3:
    use_test=list(range(30000,80001))
if cv_fold==4:
    use_test=list(range(30001))+list(range(80000,102277))
if cv_fold==5:
    use_test=list(range(12000))+list(range(24000,36000))+list(range(48000,60000))+list(range(72000,90000))
print (len(use_test),use_test[0],use_test[-1])

train_embed=np.load('../data/train_embed.npy')
test_embed=np.load('../data/test_embed.npy')[use_test]
labels=np.load('../data/labels.npy')
labels-=1
slabel=np.zeros((labels.shape[0],19))
slabel[np.arange(labels.shape[0]),labels]=1.0
np.random.seed(2018)

r1=(np.random.uniform(0,1,train_embed.shape[0])*5).astype(np.int32)
filter_t=(r1!=(cv_fold-1))
filter_v=~filter_t
x_train , y_train = train_embed[filter_t],slabel[filter_t]
x_val  ,  y_val  = train_embed[filter_v],slabel[filter_v]

test_pred_labels=np.load('../stacking/stacking.npy')[use_test]
'''在数据中加入一半的测试集，这是使用了虚假标签的方法！由于正确率大概是80%左右，
可以想象，一共15000个训练样本，正确标记样本有14000,不到10%的噪声而已！这样做是为了增加样本的多样性，防止过拟合。可以见到更多的词语组合。
在CNN和RNN中可能更加有效果。'''
x_train=np.concatenate((x_train,test_embed),axis=0)
y_train=np.concatenate((y_train,test_pred_labels))
print (x_train.shape,y_train.shape)


import random
import gc
r2=list(range(x_train.shape[0]))

saver = tf.train.Saver()
lastacc=0
lastloss=99999
learning_rate=1e-3
embed_rate=1e-4
finetune=False
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(50):
        ite=0
        random.shuffle(r2) 
        while(ite<x_train.shape[0]):
            gc.collect()
            global_step=sess.run(ss.global_steps)
            feed={ss.x:x_train[r2[ite:ite+arg.batch_size],:], ss.y:y_train[r2[ite:ite+arg.batch_size]],
                  ss.training:True,ss.lr:learning_rate,ss.lr_embed:embed_rate,ss.lamda:1e-4,
                  ss.keep_prob:arg.drop_keep_prob}
            ite+=arg.batch_size
            if finetune:
                sess.run([ss.train_op],feed_dict=feed)
            else:
                sess.run([ss.train_step2],feed_dict=feed)
            if (ite//arg.batch_size)%50==0:
                print (sess.run([ss.acc,ss.losses,ss.loss_add_reg,ss.global_steps],feed_dict=feed),learning_rate,finetune)

        ite=0
        mypred=[]
        myloss=0
        while(ite<x_val.shape[0]):
            gc.collect()
            feed={ss.x:x_val[ite:ite+arg.batch_size,:],ss.y:y_val[ite:ite+arg.batch_size],
                    ss.training:False,
                    ss.keep_prob:1.0}
            pred,loss=sess.run([ss.logit,ss.losses],feed_dict=feed)
            mypred.extend(list(np.argmax(pred,1)+1))
            myloss+=loss*x_val[ite:ite+arg.batch_size,:].shape[0]
            ite+=arg.batch_size
        myloss/=x_val.shape[0]
        acc=np.mean(np.array(mypred)==(np.argmax(y_val,1)+1))
        print (acc,myloss)
        if  myloss<lastloss:
            saver.save(sess,"../data/checkpoint/cnn%d.ckpt"%cv_fold)
            if myloss<lastloss:
                lastloss=myloss
        else:
            if finetune:
                learning_rate/=1.25
                embed_rate/=1.25
            else:
                learning_rate/=2
                embed_rate/=2
            if learning_rate<2e-4:
                finetune=True 
            if learning_rate<6e-5:
                break



saver = tf.train.Saver()
import gc
with tf.Session() as sess:
    saver.restore(sess, "../data/checkpoint/cnn%d.ckpt"%cv_fold)
    ite=0
    mypred=[]
    myloss=0
    x_val_proba=np.zeros((x_val.shape[0],19))
    while(ite<x_val.shape[0]):
        gc.collect()
        feed={ss.x:x_val[ite:ite+arg.batch_size,:],ss.y:y_val[ite:ite+arg.batch_size],
                ss.training:False,
                ss.keep_prob:1.0}
        
        pred,loss,proba=sess.run([ss.logit,ss.spareloss,ss.proba],feed_dict=feed)
        mypred.extend(list(np.argmax(pred,1)+1))
        myloss+=loss*x_val[ite:ite+arg.batch_size,:].shape[0]
        x_val_proba[ite:ite+arg.batch_size,:]=np.array(proba)
        ite+=arg.batch_size        
    myloss/=x_val.shape[0]
    acc=np.mean(np.array(mypred)==np.argmax(y_val,1)+1)
    print (cv_fold,acc,myloss)
    np.save('../stacking/cnn/val_cnn_%d.npy'%cv_fold,x_val_proba)
    
test_embed=np.load('../data/test_embed.npy')
import gc
gc.collect()
print ("test")
saver = tf.train.Saver()
mypred=[]
with tf.Session() as sess:
    saver.restore(sess, "../data/checkpoint/cnn%d.ckpt"%cv_fold)
    ite=0
    x_test_proba=np.zeros((test_embed.shape[0],19))
    while(ite<test_embed.shape[0]):
        gc.collect()
        feed={ss.x:test_embed[ite:ite+arg.batch_size,:],
                ss.training:False,
                ss.keep_prob:1.0}
        proba,logit=sess.run([ss.proba,ss.logit],feed_dict=feed)
        x_test_proba[ite:ite+arg.batch_size,:]=np.array(proba)
        mypred.extend(list(np.argmax(logit,1)+1))
        ite+=arg.batch_size
    np.save('../stacking/cnn/test_cnn_%d.npy'%cv_fold,x_test_proba)




    
    
    
    
    
    
    
    
    
    
