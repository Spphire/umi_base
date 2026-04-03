import json
import pathlib

import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _save_action_plot(gt: np.ndarray, pred: np.ndarray, path: pathlib.Path):
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
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_obs_images(obs: dict, output_dir: pathlib.Path):
    for key, value in obs.items():
        if not torch.is_tensor(value) or value.ndim != 5:
            continue
        img = value[0, -1].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        plt.imsave(output_dir / f"{key}.png", img)


def _save_attention_plots(summary: dict, output_dir: pathlib.Path):
    aggregate = summary.get("aggregate_mean_by_key", {})
    if len(aggregate) == 0:
        return

    keys = list(aggregate.keys())
    values = [aggregate[key] for key in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, values)
    ax.set_ylabel("mean attention mass")
    ax.set_title("Cross-Attention Mean By Key")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_attention_mean_by_key.png", dpi=150)
    plt.close(fig)

    per_step = summary.get("per_step", [])
    if len(per_step) == 0:
        return

    step_labels = [str(item["diffusion_timestep"]) for item in per_step]
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in keys:
        series = [item["mean_attention_by_key"].get(key, 0.0) for item in per_step]
        ax.plot(step_labels, series, marker="o", linewidth=1, label=key)
    ax.set_xlabel("diffusion timestep")
    ax.set_ylabel("mean attention mass")
    ax.set_title("Cross-Attention By Diffusion Step")
    ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_attention_by_step.png", dpi=150)
    plt.close(fig)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
    config_name="train_diffusion_transformer_timm_single_frame_workspace",
)
def main(cfg: OmegaConf):
    if "ckpt_path" not in cfg or not cfg.ckpt_path:
        raise ValueError("ckpt_path is required. Example: +ckpt_path=path/to/checkpoints/XXXX.ckpt")

    ckpt_path = pathlib.Path(cfg.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    runtime_cfg = OmegaConf.create(payload["cfg"])
    if "eval" in cfg:
        runtime_cfg.eval = cfg.eval

    cls = hydra.utils.get_class(runtime_cfg._target_)
    workspace: BaseWorkspace = cls(runtime_cfg)
    state_dicts = payload.get("state_dicts", {})
    if "model" in state_dicts:
        workspace.model.load_state_dict(state_dicts["model"])
    if "ema_model" in state_dicts and getattr(workspace, "ema_model", None) is not None:
        workspace.ema_model.load_state_dict(state_dicts["ema_model"])

    policy: BaseImagePolicy = workspace.model
    if runtime_cfg.training.use_ema and hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)

    dataset = hydra.utils.instantiate(runtime_cfg.task.dataset)
    eval_cfg = runtime_cfg.get("eval", {})
    split = str(eval_cfg.get("split", "train")).lower()
    if split == "val":
        dataset = dataset.get_validation_dataset()
    elif split != "train":
        raise ValueError(f"Unsupported eval split: {split}")

    num_samples = int(eval_cfg.get("num_samples", 10))
    start_index = int(eval_cfg.get("start_index", 0))
    output_dir = pathlib.Path(eval_cfg.get("output_dir", "output_images/transformer_cross_attention"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    saved = 0
    for sample_idx, batch in enumerate(dataloader):
        if sample_idx < start_index:
            continue
        if saved >= num_samples:
            break

        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        obs = batch["obs"]
        gt_action = batch["action"]

        with torch.no_grad():
            result = policy.predict_action(obs, return_attention=True)

        pred_action = result["action_pred"]
        attention_summary = result.get("cross_attention_summary")

        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        gt = gt_action[0].detach().cpu().numpy()
        pred = pred_action[0].detach().cpu().numpy()
        _save_action_plot(gt, pred, sample_dir / "actions.png")
        _save_obs_images(obs, sample_dir)

        if attention_summary is not None:
            with (sample_dir / "cross_attention_summary.json").open("w", encoding="utf-8") as f:
                json.dump(attention_summary, f, indent=2)
            _save_attention_plots(attention_summary, sample_dir)

        saved += 1

    print(f"Saved transformer cross-attention visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
