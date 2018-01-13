#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/posTagger.py --mode LSTM --num_epochs 200 --batch_size 16 --hidden_size 256 \
 --char_dim 30 --num_filters 30 \
 --learning_rate 0.1 --decay_rate 0.05 --schedule 10 --gamma 0.0 \
 --dropout std --unk_replace 0.0 --p 0.5 \
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" 
