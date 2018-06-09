import unittest
import tempfile
import shutil
import os
from neuronlp2.io.alphabet import Alphabet
import random, string
from neuronlp2.io.reader import CoNLL03Reader
import neuronlp2.io.conll03_data as conll03_data

def random_name(num_char):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(num_char))

TRAIN_DATA = \
"""1 CRICKET NNP I-NP O
2 - : O O
3 LEICESTERSHIRE NNP I-NP B-ORG
4 TAKE NNP I-NP O
5 OVER IN I-PP O
6 AT NNP I-NP O
7 TOP NNP I-NP O
8 AFTER NNP I-NP O
9 INNINGS NNP I-NP O
10 VICTORY NN I-NP O
11 . . O O

1 KANSAS NNP I-NP B-ORG
2 CITY NNP I-NP I-ORG
3 at NNP I-NP O
4 DETROIT NNP I-NP B-LOC
5 2.73 CD I-NP O
6 . . O O

1 -DOCSTART- -X- O O"""

DEV_DATA = \
"""2 " " O O
2 The DT I-NP O
3 market NN I-NP O
4 steady JJ I-ADJP O
"""

EMBEDD_DICT = {"KANSAS": 0, "city": 1, "market": 2}

class CoNLL03DataTest(unittest.TestCase):
    def setUp(self):
        self.alphabet_dir_path = os.path.join(tempfile.gettempdir(), random_name(8))
        if os.path.exists(self.alphabet_dir_path):
            raise IOError(self.alphabet_dir_path + " already exists!")

        self.conll03_train_file_path = os.path.join(tempfile.gettempdir(), random_name(8))
        if os.path.exists(self.conll03_train_file_path):
            raise IOError(self.conll03_train_file_path + " already exists!")
        self.write_train_data(self.conll03_train_file_path)

        self.conll03_dev_file_path = os.path.join(tempfile.gettempdir(), random_name(8))
        if os.path.exists(self.conll03_dev_file_path):
            raise IOError(self.conll03_dev_file_path + " already exists!")
        self.write_dev_data(self.conll03_dev_file_path)

    def tearDown(self):
        if os.path.exists(self.alphabet_dir_path):
            shutil.rmtree(self.alphabet_dir_path)
        if os.path.exists(self.conll03_train_file_path):
            os.unlink(self.conll03_train_file_path)
        if os.path.exists(self.conll03_dev_file_path):
            os.unlink(self.conll03_dev_file_path)

    def write_train_data(self, file_path):
        with open(file_path, "w") as file:
            file.write(TRAIN_DATA)

    def write_dev_data(self, file_path):
        with open(file_path, "w") as file:
            file.write(DEV_DATA)

    def test_train_no_dict(self):
        alphabet = Alphabet('test', use_default_value=True)
        word_alphabet, character_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet = \
            conll03_data.create_alphabets(self.alphabet_dir_path, self.conll03_train_file_path)
        # 4 words: <_UNK>, PAD, '.'
        self.assertEquals(word_alphabet.size(), 3)
        self.assertEquals(word_alphabet.singleton_size(), 0)
        # <_UNK>, _PAD_CHAR, -, ., ACDEFGHIKLNOPRSTVY, at (20 alphabet chars), 237 (3 digits)
        self.assertEquals(character_alphabet.size(), 27)
        # _PAD_POS, ., :, -X-, CD, IN, NN, NNP
        self.assertEquals(pos_alphabet.size(), 8)
        # _PAD_CHUNK, O, I-NP, I-PP
        self.assertEquals(chunk_alphabet.size(), 4)
        # _PAD_NER, O, B-ORG, I-ORG, B-LOC
        self.assertEquals(ner_alphabet.size(), 5)

    def test_train_with_dict(self):
        alphabet = Alphabet('test', use_default_value=True)
        word_alphabet, character_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet = \
            conll03_data.create_alphabets(self.alphabet_dir_path, self.conll03_train_file_path, embedd_dict=EMBEDD_DICT)
        # 4 words: <_UNK>, PAD, , '.'; KANSAS, city (are singletons but are in embedd dict)
        # word_alphabet contains non-rare words + rare words but in embedd dict
        self.assertEquals(word_alphabet.size(), 5)
        # KANSAS and city are both singletons-in-vocab and in embedd dict
        self.assertEquals(word_alphabet.singleton_size(), 2)
        # <_UNK>, _PAD_CHAR, -, ., ACDEFGHIKLNOPRSTVY, at (20 alphabet chars), 237 (3 digits)
        self.assertEquals(character_alphabet.size(), 27)
        # _PAD_POS, ., :, -X-, CD, IN, NN, NNP
        self.assertEquals(pos_alphabet.size(), 8)
        # _PAD_CHUNK, O, I-NP, I-PP
        self.assertEquals(chunk_alphabet.size(), 4)
        # _PAD_NER, O, B-ORG, I-ORG, B-LOC
        self.assertEquals(ner_alphabet.size(), 5)

    def test_train_with_dict_and_dev(self):
        alphabet = Alphabet('test', use_default_value=True)
        word_alphabet, character_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet = \
            conll03_data.create_alphabets(self.alphabet_dir_path, self.conll03_train_file_path,
                                          data_paths=[self.conll03_dev_file_path], embedd_dict=EMBEDD_DICT)
        # 4 words: <_UNK>, PAD, , '.'; KANSAS, city (are singletons but are in embedd dict)
        # word_alphabet contains non-rare words + rare words but in embedd dict + 1 in the dev
        self.assertEquals(word_alphabet.size(), 6)
        # KANSAS and city are both singletons-in-vocab and in embedd dict
        self.assertEquals(word_alphabet.singleton_size(), 2)
        # <_UNK>, _PAD_CHAR, -, ., ACDEFGHIKLNOPRSTVY, at (20 alphabet chars), 237 (3 digits)
        self.assertEquals(character_alphabet.size(), 27)
        # _PAD_POS, ., :, -X-, CD, IN, NN, NNP + DT, JJ, " in dev set
        self.assertEquals(pos_alphabet.size(), 11)
        # _PAD_CHUNK, O, I-NP, I-PP + I-ADJP in dev set
        self.assertEquals(chunk_alphabet.size(), 5)
        # _PAD_NER, O, B-ORG, I-ORG, B-LOC
        self.assertEquals(ner_alphabet.size(), 5)
