__author__ = 'max'


from instance import DependencyInstance
from instance import Sentence
import conll_data


class CoNLLReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        lines = []
        line = self.__source_file.readline()
        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            line = line.decode('utf-8')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        words.append(conll_data.ROOT)
        word_ids.append(self.__word_alphabet.get_index(conll_data.ROOT))
        char_seqs.append([conll_data.ROOT_CHAR, ])
        char_id_seqs.append([self.__char_alphabet.get_index(conll_data.ROOT_CHAR), ])
        postags.append(conll_data.ROOT_POS)
        pos_ids.append(self.__pos_alphabet.get_index(conll_data.ROOT_POS))
        types.append(conll_data.ROOT_TYPE)
        type_ids.append(self.__type_alphabet.get_index(conll_data.ROOT_TYPE))
        heads.append(-1)

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > conll_data.MAX_CHAR_LENGTH:
                chars = chars[:conll_data.MAX_CHAR_LENGTH]
                char_ids = char_ids[:conll_data.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = conll_data.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[4]
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads, types, type_ids)
