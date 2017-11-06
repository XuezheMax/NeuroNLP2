#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python examples/NERCRF.py --mode LSTM --num_epochs 200 --batch_size 8 --hidden_size 256 --num_filters 30 \
 --learning_rate 0.015 --decay_rate 0.05 --schedule 5 --gamma 0.0 --dropout std --p 0.5 \
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" \
 --train "data/conll2003/english/tmp.bioes.conll" --dev "data/conll2003/english/tmp.test.bioes.conll" --test "data/conll2003/english/tmp.test.bioes.conll"
