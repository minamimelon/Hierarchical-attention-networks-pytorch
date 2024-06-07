export CUDA_VISIBLE_DEVICES=8
python3 train.py --word2vec_path  glove.6B.100d.txt --log_path tensorboard/han_voc_100 --saved_path trained_models_100
python3 train.py --word2vec_path  glove.6B.200d.txt --log_path tensorboard/han_voc_200 --saved_path trained_models_200
python3 train.py --word2vec_path  glove.6B.300d.txt --log_path tensorboard/han_voc_300 --saved_path trained_models_300
