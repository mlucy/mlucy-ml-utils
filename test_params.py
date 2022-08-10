import params

import unittest
import torch
from torch import nn
from pprint import pprint

class _TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(1, 2, 3)
        self.gn1 = nn.GroupNorm(1, 2)
        self.seq = nn.Sequential(
            nn.Linear(1, 2),
            nn.SiLU(),
            nn.Linear(2, 3),
        )

class Test(unittest.TestCase):
    def test_group_params(self):
        tm = _TestModule()
        res = params.group_params(tm, params.adamw_easy_grouper, output='names')
        assert res == {
            'decay': ['seq.0.weight', 'seq.2.weight'],
            'no_decay': ['ln1.weight',
                         'ln1.bias',
                         'gn1.weight',
                         'gn1.bias',
                         'seq.0.bias',
                         'seq.2.bias'],
        }

if __name__ == '__main__':
    unittest.main()
