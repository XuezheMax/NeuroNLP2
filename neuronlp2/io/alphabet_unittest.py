import unittest

from neuronlp2.io import alphabet


class AlphabetTest(unittest.TestCase):
    def test_empty(self):
        vocab = alphabet.Alphabet('test', True)
        self.assertEqual(vocab.get_instance(0), alphabet.DEFAULT_VALUE)
        vocab = alphabet.Alphabet('test', False)
        self.assertRaises(IndexError, vocab.get_instance, 0)

    def test_add_with_default(self):
        vocab = alphabet.Alphabet('test', True)
        word1 = 'word1'
        vocab.add(word1)
        self.assertEqual(vocab.get_index(word1), 1)
        word2 = 'word2'
        vocab.add(word2)
        self.assertEqual(vocab.get_index(word2), 2)

    def test_add_with_no_default(self):
        vocab = alphabet.Alphabet('test')
        word1 = 'word1'
        vocab.add(word1)
        self.assertEqual(vocab.get_index(word1), 0)
        word2 = 'word2'
        vocab.add(word2)
        self.assertEqual(vocab.get_index(word2), 1)

    def test_add_duplicate(self):
        vocab = alphabet.Alphabet('test')
        word1 = 'word1'
        vocab.add(word1)
        self.assertEqual(vocab.get_index(word1), 0)
        vocab.add(word1)
        self.assertEqual(vocab.get_index(word1), 0)

    def test_singletons(self):
        vocab = alphabet.Alphabet('test', singleton=True)
        self.assertEquals(vocab.singleton_size(), 0)
        word1 = 'word1'
        vocab.add(word1)
        vocab.add_singleton(vocab.get_index(word1))
        self.assertEquals(vocab.singleton_size(), 1)
        self.assertTrue(vocab.is_singleton(vocab.get_index(word1)))
        word2 = 'word2'
        vocab.add(word2)
        self.assertFalse(vocab.is_singleton(vocab.get_index(word2)))
