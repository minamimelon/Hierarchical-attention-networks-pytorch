export CUDA_VISIBLE_DEVICES=2
python3 train.py --word2vec_path  glove.840B.300d.txt --log_path tensorboard/han_voc_w100_s50 --saved_path trained_models_w100_s50 --word_hidden_size 100 --sent_hidden_size 50
python3 train.py --word2vec_path  glove.840B.300d.txt --log_path tensorboard/han_voc_w25_s50 --saved_path trained_models_w25_s50 --word_hidden_size 25 --sent_hidden_size 50
python3 train.py --word2vec_path  glove.840B.300d.txt --log_path tensorboard/han_voc_w25_s50 --saved_path trained_models_w75_s50 --word_hidden_size 75 --sent_hidden_size 50
python3 train.py --word2vec_path  glove.840B.300d.txt --log_path tensorboard/han_voc_w75_s50 --saved_path trained_models_w75_s50 --word_hidden_size 75 --sent_hidden_size 50
