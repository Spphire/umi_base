import os
import pathlib
import pickle

import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
    config_name="train_diffusion_unet_timm_single_frame_workspace",
)
def main(cfg: OmegaConf):
    #cfg.ckpt_path='/mnt/data/shenyibo/workspace/umi_base/data/outputs/2026.02.03/10.09.10_train_diffusion_unet_timm_q3_shop_bagging_0202_100/checkpoints/1190.ckpt'
    if "ckpt_path" not in cfg or not cfg.ckpt_path:
        raise ValueError("ckpt_path is required. Example: +ckpt_path=path/to/checkpoints/XXXX.ckpt")

    ckpt_path = pathlib.Path(cfg.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if "diffusion" in cfg.name:
        policy: BaseImagePolicy = workspace.model
        if cfg.training.use_ema and hasattr(workspace, "ema_model"):
            policy = workspace.ema_model
    else:
        raise NotImplementedError("Only diffusion policies are supported in this script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)

    dataset = hydra.utils.instantiate(cfg.task.dataset)

    eval_cfg = cfg.get("eval", {})
    num_samples = int(eval_cfg.get("num_samples", 10))
    output_dir = pathlib.Path(eval_cfg.get("output_dir", "output_images/trainset_visualization"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for sample_idx, batch in enumerate(dataloader):
        if sample_idx >= num_samples:
            break

        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        obs = batch["obs"]
        gt_action = batch["action"]

        #device = next(policy.parameters()).device
        #obs = dict_apply(obs, lambda x: x.to(device) if torch.is_tensor(x) else x)

        with torch.no_grad():
            result = policy.predict_action(obs)

        pred_action = result["action_pred"]

        gt = gt_action[0].detach().cpu().numpy()  # (T, D)
        pred = pred_action[0].detach().cpu().numpy()  # (T, D)

        # Plot action curves
        num_dims = gt.shape[-1]
        fig, axes = plt.subplots(num_dims, 1, figsize=(8, 2 * num_dims), sharex=True)
        if num_dims == 1:
            axes = [axes]
        for dim in range(num_dims):
            axes[dim].plot(gt[:, dim], label="gt", linewidth=1)
            axes[dim].plot(pred[:, dim], label="pred", linewidth=1)
            axes[dim].set_ylabel(f"dim_{dim}")
            axes[dim].legend(loc="upper right")
        axes[-1].set_xlabel("timestep")
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{sample_idx:03d}_actions.png", dpi=150)
        plt.close(fig)

        # Save one image per rgb key (last frame)
        for key, value in obs.items():
            #print(f"{key}: {value.shape}")
            if value.ndim == 5:  # (B, T, C, H, W)
                img = value[0, -1].detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = np.clip(img, 0.0, 1.0)
                plt.imsave(output_dir / f"sample_{sample_idx:03d}_{key}.png", img)

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
