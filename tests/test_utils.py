import re
from flexa._utils import *
import unittest

import numpy as np

class Test_utils(unittest.TestCase):
    def test_sortkey(self):
        key = (1, 5)
        self.assertEqual(sortkey(key), key)
        self.assertEqual(sortkey(key[::-1]), key)

        i = key[0]
        j = key[1]
        self.assertEqual(sortkey(i, j), key)
        self.assertEqual(sortkey(j, i), key)
    
    def test_pairs(self):
        self.assertRaises(ValueError, pairs, range(2))
        self.assertEqual(pairs(range(3)), [(0, 1), (1, 2), (2, 0)])

        n = 10 
        self.assertEqual(pairs(range(n)), [(i, (i + 1) % n) for i in range(n)])
    
    def test_reindex_list_of_lists(self):
        data = [[4, 6], [1, 333, 7]]
        self.assertEqual(reindex_list_of_lists(data), [[1, 2], [0, 4, 3]])
    
        keep = [1, 4, 6, 7]
        self.assertEqual(reindex_list_of_lists(data, keep), 
            [[1, 2], [0, -1, 3]])
    
    def test_picklify(self):
        base = 'hello'
        expected = base + '.p'
        self.assertEqual(picklify(base), expected)
        self.assertEqual(picklify(expected), expected)


if __name__ == '__main__':
    unittest.main()