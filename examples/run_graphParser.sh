#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/GraphParser.py --mode LSTM --num_epochs 200 --batch_size 64 --hidden_size 256 --num_layers 3 --num_filters 50 --tag_space 256 \
 --objective cross_entropy --learning_rate 0.1 --decay_rate 0.05 --schedule 5 --gamma 0.0 --dropout std --p 0.5 --biaffine \
 --word_embedding glove --word_path "data/glove/glove.6B/glove.6B.100d.gz" --char_embedding random \
 --punctuation '$(' '.' '``' "''" ':' ',' \
 --train "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.train.conll" \
 --dev "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.dev.conll" \
 --test "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.test.conll"
