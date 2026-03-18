import unittest

import numpy as np

from diffusion_policy.common.structured_sampling_index import (
    build_joint_ranked_candidates,
    compute_future_action_features,
    pairwise_l2_knn,
)


class TestStructuredSamplingIndex(unittest.TestCase):
    def test_future_action_features_padding(self):
        actions = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=np.float32,
        )
        feat = compute_future_action_features(actions, horizon=2)
        self.assertEqual(feat.shape, (3, 4))
        np.testing.assert_allclose(feat[0], np.array([1, 10, 2, 20], dtype=np.float32))
        np.testing.assert_allclose(feat[2], np.array([3, 30, 3, 30], dtype=np.float32))

    def test_pairwise_knn_basic(self):
        x = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
        idx, dist = pairwise_l2_knn(x, k=2, exclude_self=True, chunk_size=2)
        self.assertEqual(idx.shape, (3, 2))
        self.assertEqual(dist.shape, (3, 2))
        np.testing.assert_array_equal(idx[0], np.array([1, 2]))
        np.testing.assert_allclose(dist[0], np.array([1.0, 3.0], dtype=np.float32), atol=1e-6)

    def test_pairwise_knn_singleton(self):
        x = np.array([[5.0, 1.0]], dtype=np.float32)
        idx, dist = pairwise_l2_knn(x, k=3, exclude_self=True)
        np.testing.assert_array_equal(idx, np.array([[-1, -1, -1]], dtype=np.int64))
        self.assertTrue(np.isinf(dist).all())

    def test_joint_ranked_candidates_alpha_beta(self):
        # anchor 0 has two wrist-neighbors: 1,2
        wrist_knn = np.array(
            [
                [1, 2],
                [0, 2],
                [0, 1],
            ],
            dtype=np.int64,
        )
        head = np.array(
            [
                [0.0, 0.0],  # anchor
                [10.0, 0.0], # head-far
                [1.0, 0.0],  # head-near
            ],
            dtype=np.float32,
        )
        act = np.array(
            [
                [0.0, 0.0],  # anchor
                [0.0, 0.0],  # action-near
                [20.0, 0.0], # action-far
            ],
            dtype=np.float32,
        )

        idx_h, score_h, _, _ = build_joint_ranked_candidates(
            wrist_knn, head, act, alpha=1.0, beta=0.0, top_m=1
        )
        self.assertEqual(int(idx_h[0, 0]), 1)  # choose head-far
        self.assertTrue(np.isfinite(score_h[0, 0]))

        idx_a, score_a, _, _ = build_joint_ranked_candidates(
            wrist_knn, head, act, alpha=0.0, beta=1.0, top_m=1
        )
        self.assertEqual(int(idx_a[0, 0]), 2)  # choose action-far
        self.assertTrue(np.isfinite(score_a[0, 0]))


if __name__ == "__main__":
    unittest.main()

