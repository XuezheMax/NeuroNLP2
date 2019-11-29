# NeuroNLP2
Deep neural models for core NLP tasks based on Pytorch(version 2)

This is the code we used in the following papers
>[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.cs.cmu.edu/~xuezhem/publications/P16-1101.pdf)

>Xuezhe Ma, Eduard Hovy

>ACL 2016

>[Neural Probabilistic Model for Non-projective MST Parsing](http://www.cs.cmu.edu/~xuezhem/publications/IJCNLP2017.pdf)

>Xuezhe Ma, Eduard Hovy

>IJCNLP 2017

>[Stack-Pointer Networks for Dependency Parsing](https://arxiv.org/pdf/1805.01087.pdf)

>Xuezhe Ma, Zecong Hu, Jingzhou Liu, Nanyun Peng, Graham Neubig and Eduard Hovy

>ACL 2018

It also includes the re-implementation of the Stanford Deep BiAffine Parser:
>[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)

>Timothy Dozat, Christopher D. Manning

>ICLR 2017

## Updates
1. Upgraded the code to support PyTorch 1.3 and Python 3.6
2. Re-factored code to better organization
3. Implemented the batch version of Stack-Pointer Parser decoding algorithm, about 50 times faster!

## Requirements

Python 3.6, PyTorch >=1.3.1, Gensim >= 0.12.0

## Data format
For the data format used in our implementation, please read this [issue](https://github.com/XuezheMax/NeuroNLP2/issues/9).

## Running the experiments
First to the experiments folder:

    cd experiments
### Sequence labeling
To train a CRF POS tagger of PTB WSJ corpus, 

    ./scripts/run_pos_wsj.sh
where the arguments for ```train/dev/test``` data, together with the pretrained word embedding should be setup.

To train a NER model on CoNLL-2003 English data set,

    ./scripts/run_ner_conll03.sh

### Dependency Parsing
To train a Stack-Pointer parser, simply run

    ./scripts/run_stackptr.sh
Remeber to setup the paths for data and embeddings.

To train a Deep BiAffine parser, simply run

    ./scripts/run_deepbiaf.sh
Again, remember to setup the paths for data and embeddings.

To train a Neural MST parser, 

    ./scripts/run_neuromst.sh
