__author__ = 'max'


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
                w = self.__word_alphabet.get_instance(word[i, j])
                p = self.__pos_alphabet.get_instance(pos[i, j])
                ch = self.__chunk_alphabet.get_instance(chunk[i, j])
                tgt = self.__ner_alphabet.get_instance(targets[i, j])
                pred = self.__ner_alphabet.get_instance(predictions[i, j])
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, p, ch, tgt, pred))
            self.__source_file.write('\n')


class POSWriter(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, predictions, targets, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                w = self.__word_alphabet.get_instance(word[i, j])
                pred = self.__pos_alphabet.get_instance(predictions[i, j])
                tgt = self.__pos_alphabet.get_instance(targets[i, j])
                self.__source_file.write('%d\t%s\t_\t%s\t%s\n' % (j, w, tgt, pred))
            self.__source_file.write('\n')


class CoNLLXWriter(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, head, type, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                w = self.__word_alphabet.get_instance(word[i, j])
                p = self.__pos_alphabet.get_instance(pos[i, j])
                t = self.__type_alphabet.get_instance(type[i, j])
                h = head[i, j]
                self.__source_file.write('%d\t%s\t_\t_\t%s\t_\t%d\t%s\n' % (j, w, p, h, t))
            self.__source_file.write('\n')
