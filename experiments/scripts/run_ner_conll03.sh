#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ner.py --config configs/ner/conll03.json --num_epochs 400 --batch_size 16 \
 --loss_type sentence --optim sgd --learning_rate 0.01 --lr_decay 0.99999 --grad_clip 0.0 --warmup_steps 100 --weight_decay 0.0 --unk_replace 0.0 \
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --model_path "models/ner/conll03" \
 --train "data/conll2003/english/eng.train.bioes.conll" --dev "data/conll2003/english/eng.dev.bioes.conll" --test "data/conll2003/english/eng.test.bioes.conll"
