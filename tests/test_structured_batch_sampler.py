import unittest

import numpy as np

from diffusion_policy.common.structured_batch_sampler import StructuredCoverageBatchSampler


class TestStructuredBatchSampler(unittest.TestCase):
    def test_full_coverage_no_drop_last(self):
        n = 10
        bsz = 4
        sampler = StructuredCoverageBatchSampler(
            num_samples=n,
            batch_size=bsz,
            candidate_indices=None,
            structured_ratio=0.5,
            shuffle=False,
            drop_last=False,
            seed=0,
        )
        batches = list(iter(sampler))
        flat = [x for batch in batches for x in batch]
        self.assertEqual(len(flat), n)
        self.assertEqual(len(set(flat)), n)
        self.assertEqual(sorted(flat), list(range(n)))

    def test_drop_last(self):
        n = 10
        bsz = 4
        sampler = StructuredCoverageBatchSampler(
            num_samples=n,
            batch_size=bsz,
            candidate_indices=None,
            structured_ratio=0.5,
            shuffle=False,
            drop_last=True,
            seed=0,
        )
        batches = list(iter(sampler))
        self.assertEqual(len(batches), n // bsz)
        flat = [x for batch in batches for x in batch]
        self.assertEqual(len(flat), (n // bsz) * bsz)
        self.assertEqual(len(set(flat)), len(flat))

    def test_structured_preference(self):
        n = 8
        bsz = 4
        cand = np.full((n, 4), -1, dtype=np.int64)
        cand[0] = np.array([3, 2, 7, 6], dtype=np.int64)
        sampler = StructuredCoverageBatchSampler(
            num_samples=n,
            batch_size=bsz,
            candidate_indices=cand,
            structured_ratio=0.75,  # target_structured = 3
            shuffle=False,
            drop_last=False,
            seed=0,
        )
        batches = list(iter(sampler))
        first = batches[0]
        # Anchor 0 should pull structured candidates 3 and 2 first.
        self.assertEqual(first[0], 0)
        self.assertIn(3, first[:3])
        self.assertIn(2, first[:3])

    def test_set_epoch_changes_order_with_shuffle(self):
        n = 50
        bsz = 5
        sampler = StructuredCoverageBatchSampler(
            num_samples=n,
            batch_size=bsz,
            candidate_indices=None,
            structured_ratio=0.5,
            shuffle=True,
            drop_last=False,
            seed=123,
        )
        sampler.set_epoch(0)
        batches0 = list(iter(sampler))
        sampler.set_epoch(1)
        batches1 = list(iter(sampler))
        self.assertNotEqual(batches0[0], batches1[0])


if __name__ == "__main__":
    unittest.main()

