CUDA_VISIBLE_DEVICES=0 python examples/posTagger.py --mode LSTM --num_epochs 200 --batch_size 16 --hidden_size 128 --num_filters 30 \
 --learning_rate 0.1 --decay_rate 0.05 --schedule 20 30 40 50 60 70 80 90 100 120 140 160 180 --gamma 0.0 --dropout variational --p 0.5 \
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" 
