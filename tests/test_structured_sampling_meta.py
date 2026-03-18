import tempfile
import unittest
from pathlib import Path

from diffusion_policy.common.structured_sampling_meta import (
    build_meta,
    is_meta_compatible,
    load_meta,
    save_meta,
)


class TestStructuredSamplingMeta(unittest.TestCase):
    def _cfg(self):
        return {
            "name": "train_diffusion_unet_timm",
            "task_name": "q3_mouse",
            "policy": {
                "obs_encoder": {
                    "_target_": "diffusion_policy.model.vision.timm_obs_encoder.TimmObsEncoder",
                    "model_name": "vit_base_patch16_dinov3.lvd1689m",
                    "pretrained": True,
                }
            },
        }

    def test_meta_roundtrip_and_compatibility(self):
        cfg = self._cfg()
        sampler_sig = {"alpha": 0.5, "beta": 0.5, "k": 64}
        meta_a = build_meta(cfg, dataset_signature="dataset_hash_a", sampler_signature=sampler_sig)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "meta.json"
            save_meta(str(p), meta_a)
            loaded = load_meta(str(p))

        ok, reason = is_meta_compatible(loaded, meta_a)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_encoder_mismatch_detected(self):
        cfg_a = self._cfg()
        cfg_b = self._cfg()
        cfg_b["policy"]["obs_encoder"]["model_name"] = "vit_base_patch16_clip_224.openai"

        sampler_sig = {"alpha": 0.5, "beta": 0.5, "k": 64}
        meta_a = build_meta(cfg_a, dataset_signature="dataset_hash_a", sampler_signature=sampler_sig)
        meta_b = build_meta(cfg_b, dataset_signature="dataset_hash_a", sampler_signature=sampler_sig)

        ok, reason = is_meta_compatible(meta_a, meta_b)
        self.assertFalse(ok)
        self.assertEqual(reason, "encoder signature mismatch")


if __name__ == "__main__":
    unittest.main()

