import tempfile
import unittest

import numpy as np
import torch

from diffusion_policy.common.structured_dataloader import (
    build_train_dataloader,
    set_epoch_for_structured_sampler,
)


class TestStructuredDataloader(unittest.TestCase):
    def test_build_without_structured_sampling(self):
        ds = torch.utils.data.TensorDataset(torch.arange(10))
        cfg = {
            "batch_size": 4,
            "shuffle": False,
            "num_workers": 0,
        }
        loader, sampler = build_train_dataloader(ds, cfg, default_seed=7)
        self.assertIsNone(sampler)
        batches = list(loader)
        self.assertEqual(len(batches), 3)

    def test_build_with_structured_sampling(self):
        ds = torch.utils.data.TensorDataset(torch.arange(10))
        cand = np.full((10, 4), -1, dtype=np.int64)
        cand[0] = np.array([1, 2, 3, 4], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/cand.npy"
            np.save(path, cand)

            cfg = {
                "batch_size": 4,
                "shuffle": False,
                "drop_last": False,
                "num_workers": 0,
                "structured_sampling": {
                    "enabled": True,
                    "candidate_indices_path": path,
                    "structured_ratio": 0.75,
                    "seed": 123,
                },
            }
            loader, sampler = build_train_dataloader(ds, cfg, default_seed=7)
            self.assertIsNotNone(sampler)
            self.assertTrue(set_epoch_for_structured_sampler(loader, 2))

            flat = []
            for batch in loader:
                # TensorDataset returns tuple of tensors.
                idx = batch[0].numpy().tolist()
                flat.extend(idx)
            self.assertEqual(len(flat), 10)
            self.assertEqual(len(set(flat)), 10)


if __name__ == "__main__":
    unittest.main()

