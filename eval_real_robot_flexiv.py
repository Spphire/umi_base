# %%
import pathlib
import torch
import dill
import hydra
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import os
import psutil
import cv2

os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
cv2.setNumThreads(12)

total_cores = psutil.cpu_count()
num_cores_to_bind = 10
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
os.sched_setaffinity(0, cores_to_bind)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy', 'config')),
    config_name="train_diffusion_unet_real_image_workspace"
)
def main(cfg):
    if "ckpt_path" in cfg and cfg.ckpt_path:
        ckpt_path = str(pathlib.Path(__file__).parent.joinpath(cfg.ckpt_path))
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        if 'diffusion' in cfg.name:
            policy: BaseImagePolicy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model

            device = torch.device('cuda')
            policy.eval().to(device)

            policy.num_inference_steps = 8
            # policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        else:
            raise NotImplementedError("Non-diffusion model not implemented in flexiv mode.")

        env_runner = hydra.utils.instantiate(cfg.task.env_runner)
        env_runner.run(policy)
    else:
        env_runner = hydra.utils.instantiate(cfg.task.env_runner)
        env_runner.run()

if __name__ == '__main__':
    main()
