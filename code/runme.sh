echo "copy data"
cp ../data/train_set.csv ../team/data/member1_data
cp ../data/test_set.csv ../team/data/member1_data

echo "run team first"
cd ../team/code
./train.sh


echo "run team2"
cd ../../code

echo "data_preprocess"
python data_preprocess.py
echo "tradition feature for word"
python tradition_feat.py "word_seg"
echo "tradition feature for char"
python tradition_feat.py "article"

echo "merge team_data"
cp  ../team/data/member1_data/team_data/train_x.npy  ../stacking/team
cp  ../team/data/member1_data/team_data/test_x.npy  ../stacking/team
cp ../team/data/member1_data/article_train_tfidf_svd.npy  ../data
cp ../team/data/member1_data/article_test_tfidf_svd.npy ../data

echo "generate presudo labels"
cd ..
cd stacking
python generate_presudo_labels.py

echo "deep model"
cd ..
cd model

echo "Fast_attention"
python Fast_attention.py 1
python Fast_attention.py 2
python Fast_attention.py 3
python Fast_attention.py 4
python Fast_attention.py 5
echo "Fast_attention_withsta"
python Fast_attention_withsta.py 1
python Fast_attention_withsta.py 2
python Fast_attention_withsta.py 3
python Fast_attention_withsta.py 4
python Fast_attention_withsta.py 5

echo "Fast_attention_withsta2"
python Fast_attention_withsta2.py 1
python Fast_attention_withsta2.py 2
python Fast_attention_withsta2.py 3
python Fast_attention_withsta2.py 4
python Fast_attention_withsta2.py 5

echo "TextCNN"
python TextCNN.py 1
python TextCNN.py 2
python TextCNN.py 3
python TextCNN.py 4
python TextCNN.py 5

echo "RCNN"
python RCNN.py 1
python RCNN.py 2
python RCNN.py 3
python RCNN.py 4
python RCNN.py 5

echo "rnnpool"
python rnnpool.py 1
python rnnpool.py 2
python rnnpool.py 3
python rnnpool.py 4
python rnnpool.py 5

echo "RNN_attention"
python RNN_attention.py 1
python RNN_attention.py 2
python RNN_attention.py 3
python RNN_attention.py 4
python RNN_attention.py 5


echo "RNN_attention2"
python RNN_attention2.py 1
python RNN_attention2.py 2
python RNN_attention2.py 3
python RNN_attention2.py 4
python RNN_attention2.py 5

echo "RNN_attention_withsta"
python RNN_attention_withsta.py 1
python RNN_attention_withsta.py 2
python RNN_attention_withsta.py 3
python RNN_attention_withsta.py 4
python RNN_attention_withsta.py 5

echo "RNN_attention_withsta2"
python RNN_attention_withsta2.py 1
python RNN_attention_withsta2.py 2
python RNN_attention_withsta2.py 3
python RNN_attention_withsta2.py 4
python RNN_attention_withsta2.py 5


echo "stacking"
cd ..
cd stacking
python stack.py
