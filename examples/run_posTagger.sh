#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/posTagger.py --mode LSTM --num_epochs 200 --batch_size 16 --hidden_size 256 --num_layers 1 \
 --char_dim 30 --num_filters 30 --tag_space 256 \
 --learning_rate 0.1 --decay_rate 0.05 --schedule 10 --gamma 0.0 \
 --dropout std --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --unk_replace 0.0 \
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" 
