import unittest
from . import tutils
from .config import Config
import torch

class TestBeamSearch(unittest.TestCase):
    def test_basic(self):
        width = 4
        config = Config({}, defaults={
            'length': 5,
            'n_tokens': 10,
        })
        def forward_log_probs(batch):
            return torch.arange(config.n_tokens).reshape(1, 1, 10).tile(
                (batch.shape[0], config.length, 1))
        res, _ = tutils.beam_search(
            4, 3, torch.tensor([7, 2]), forward_log_probs, config)
        scores = torch.sum(res, dim=1)
        assert torch.equal(scores, torch.tensor([36, 35, 35, 35]))

    def test_wide(self):
        width = 20
        config = Config({}, defaults={
            'length': 5,
            'n_tokens': 10,
        })
        def forward_log_probs(batch):
            return torch.arange(config.n_tokens).reshape(1, 1, 10).tile(
                (batch.shape[0], config.length, 1))
        res, _ = tutils.beam_search(
            width, 3, torch.tensor([7, 2]), forward_log_probs, config)
        scores = torch.sum(res, dim=1)
        assert torch.equal(scores, torch.tensor(
            [36, 35, 35, 35, 34, 34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33,
             33, 33],
        ))

if __name__ == '__main__':
    unittest.main()
