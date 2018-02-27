# NeuroNLP2
Deep neural models for core NLP tasks based on Pytorch(version 2)

This is the code we used in the following papers
>[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.cs.cmu.edu/~xuezhem/publications/P16-1101.pdf)

>Xuezhe Ma, Eduard Hovy

>ACL 2016

>[Neural Probabilistic Model for Non-projective MST Parsing](http://www.cs.cmu.edu/~xuezhem/publications/IJCNLP2017.pdf)

>Xuezhe Ma, Eduard Hovy

>IJCNLP 2017

It also includes the re-implementation of the Stanford Deep BiAffine Parser:
>[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)

>Timothy Dozat, Christopher D. Manning

>ICLR 2017

## Requirements

Python 2.7, PyTorch >=0.3.0, Gensim >= 0.12.0

## Data format
For the data format used in our implementation, please read this [issue](https://github.com/XuezheMax/NeuroNLP2/issues/9).

## Running the experiments

### Sequence labeling
In the root of the repository, first make the tmp directory:

    mkdir tmp

To train a CRF POS tagger, 

    ./example/run_posCRFTagger.sh
where the arguments for ```train/dev/test``` data, together with the pretrained word embedding should be setup.

To train a NER model,

    ./example/run_ner_crf.sh

### Dependency Parsing
To train a Deep BiAffine parser, simply run

    ./example/run_graphParser.sh
Again, remember to setup the paths for data and embeddings.

To train a Neural MST parser, run the same script, but change the argument ```objective``` from ```cross_entropy``` to ```crf``` (this part is still under development).
 