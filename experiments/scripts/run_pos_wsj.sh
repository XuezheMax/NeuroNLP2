#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python pos_tagging.py --config configs/pos/wsj.json --num_epochs 400 --batch_size 32 \
 --loss_type sentence --optim sgd --learning_rate 0.01 --lr_decay 0.99999 --grad_clip 0.0 --warmup_steps 10 --weight_decay 0.0 --unk_replace 0.0 \
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --model_path "models/pos/wsj" \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" 
