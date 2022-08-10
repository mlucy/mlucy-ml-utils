from BPE import BPETokenizer

import unittest

class Test(unittest.TestCase):
    def test_basic(self):
        bs = b'aaaabbaaaabaacbbcbb'
        t = BPETokenizer(bs)
        assert [x[1] for x in t.tokens()] == ['a', 'a', 'a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'b', 'a', 'a', 'c', 'b', 'b', 'c', 'b', 'b']
        t.step()
        assert [x[1] for x in t.tokens()] == ['aa', 'aa', 'b', 'b', 'aa', 'aa', 'b', 'aa', 'c', 'b', 'b', 'c', 'b', 'b']
        t.step()
        assert [x[1] for x in t.tokens()] == ['aa', 'aa', 'bb', 'aa', 'aa', 'b', 'aa', 'c', 'bb', 'c', 'bb']
        t.step()
        assert [x[1] for x in t.tokens()] == ['aaaa', 'bb', 'aaaa', 'b', 'aa', 'c', 'bb', 'c', 'bb']
        t.step()
        assert [x[1] for x in t.tokens()] == ['aaaa', 'bb', 'aaaa', 'b', 'aa', 'cbb', 'cbb']

if __name__ == '__main__':
    unittest.main()
