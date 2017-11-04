__author__ = 'max'

from instance import DependencyInstance
import conllx_data


class CoNLL03Writer(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, chunk, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                p = self.__pos_alphabet.get_instance(pos[i, j])
                ch = self.__chunk_alphabet.get_instance(chunk[i, j])
                tgt = self.__ner_alphabet.get_instance(targets[i, j])
                pred = self.__ner_alphabet.get_instance(predictions[i, j])
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, p, ch, tgt, pred))
            self.__source_file.write('\n')
