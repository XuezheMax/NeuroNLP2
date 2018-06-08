import unittest

from neuronlp2.io import alphabet


class AlphabetTest(unittest.TestCase):
  def testInitialization(self):
    vocab = alphabet.Alphabet('test', True)
    self.assertEqual(vocab.instance2index('unknown'), 0)

  def testAdd(self):
    vocab = alphabet.Alphabet('test', True)
    vocab.add('abc')
    self.assertEqual(vocab.instance2index('abc'), 1)

if __name__ == '__main__':
  unittest.main()