#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python examples/StackPointerParser.py --mode FastLSTM --num_epochs 200 --batch_size 32 --hidden_size 512 --num_layers 3 \
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --schedule 5 --coverage 0.0 --gamma 0.0 --clip 5.0 \
 --p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --beam 5 --prior_order depth_first \
 --word_embedding glove --word_path "data/glove/glove.6B/glove.6B.100d.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.train.conll" \
 --dev "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.dev.conll" \
 --test "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.test.conll" \
 --model_path "models/parsing/stack_ptr/" --model_name 'network.pt'
