#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/analyze.py --parser stackptr --beam 5 --ordered --gpu \
 --punctuation '.' '``' "''" ':' ',' \
 --test "data/PTB3.0/PTB3.0-Stanford_dep/ptb3.0-stanford.auto.cpos.test.conll" \
 --model_path "models/parsing/stack_ptr/" --model_name 'network.pt'
