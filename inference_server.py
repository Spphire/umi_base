import os
import pathlib
import dill
import torch
import hydra
from omegaconf import OmegaConf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
from diffusion_policy.common.space_utils import pose_7d_to_4x4matrix, matrix4x4_to_pose_7d, pose_3d_9d_to_homo_matrix_batch

# Avoid OpenMP duplicate load issue (Windows common, harmless on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = FastAPI(title="UMI Simple Inference Server", version="1.0.0")

# 启动时加载模型
MODEL_CKPT = os.environ.get("MODEL_CKPT", "./latest.ckpt")
MODEL_CFG = os.environ.get("MODEL_CFG", None)

def load_policy(ckpt_path: str, cfg_yaml_path: Optional[str] = None):
    """Load policy checkpoint (workspace-based)."""
    ckpt_path = pathlib.Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading policy from: {ckpt_path}")
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")

    if cfg_yaml_path is not None:
        cfg_yaml_path = pathlib.Path(cfg_yaml_path).expanduser().resolve()
        if not cfg_yaml_path.is_file():
            raise FileNotFoundError(f"Config yaml not found: {cfg_yaml_path}")
        yaml_cfg = OmegaConf.load(str(cfg_yaml_path))
        base_cfg = OmegaConf.create(payload["cfg"])
        OmegaConf.set_struct(base_cfg, False)
        cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        print(f"  Using cfg merged from yaml: {cfg_yaml_path}")
    else:
        cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=['lr_scheduler'], include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema and hasattr(workspace, "ema_model"):
        policy = workspace.ema_model
        print("  Using EMA model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    print(f"  Policy loaded on device: {device}")
    return policy, cfg, device

policy, cfg, device = load_policy(MODEL_CKPT, MODEL_CFG)

@app.post("/v1/inference")
async def inference_endpoint(request: Request):
    try:
        data = await request.json()
        init_pose = data.get("init_pose", None)
        obs = {}
        # 预处理 base64 图片和 wrist/eye
        for k, v in data.items():
            if k.endswith("_img") and isinstance(v, str):
                import base64
                import io
                from PIL import Image, ImageOps
                img_bytes = base64.b64decode(v)
                if "wrist" in k:
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)

                    arr = arr.reshape((1, 3, 224, 224))
                    obs[k] = arr
                elif "eye" in k:
                    img = Image.open(io.BytesIO(img_bytes))
                    # pad到正方形
                    w, h = img.size
                    if w != h:
                        pad_left = pad_top = 0
                        pad_right = pad_bottom = 0
                        if w > h:
                            pad_top = (w - h) // 2
                            pad_bottom = w - h - pad_top
                        else:
                            pad_left = (h - w) // 2
                            pad_right = h - w - pad_left
                        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
                    img = img.resize((224, 224))
                    arr = np.array(img)[:,:,::-1]  # BGR to RGB
                    if arr.shape[-1] == 3:
                        arr = np.transpose(arr, (2, 0, 1))  # CHW
                    obs[k] = arr.reshape((1, 3, 224, 224))

        # add relative state
        if "left_wrist_img" in obs:
            obs['left_robot_tcp_pose'] = np.array([[0,0,0,1,0,0,0,1,0]], dtype=np.float32)  # dummy tcp pose
        if "right_wrist_img" in obs:
            obs['right_robot_tcp_pose'] = np.array([[0,0,0,1,0,0,0,1,0]], dtype=np.float32)  # dummy tcp pose
        # 转 tensor
        obs_tensor = {kk: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, (list, tuple, np.ndarray)) else v for kk, v in obs.items()}
        with torch.no_grad():
            result = policy.predict_action(obs_tensor)
        pred_full = result["action_pred"][0].detach().cpu().numpy()

        init_pose_mat_unity = pose_7d_to_4x4matrix(np.array(init_pose, dtype=np.float32)) if init_pose is not None else np.eye(4)
        flipZ = np.diag([1, 1, -1, 1])
        init_pose_mat = flipZ @ init_pose_mat_unity @ flipZ
        action_absolute_mat = init_pose_mat @ pose_3d_9d_to_homo_matrix_batch(pred_full[:,:9])
        action_absolute_mat_unity = flipZ @ action_absolute_mat @ flipZ
        pose_7d_absolute_unity = matrix4x4_to_pose_7d(action_absolute_mat_unity)

        return JSONResponse(content={"success": True, "data": pose_7d_absolute_unity.tolist()})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "umi-simple-inference-server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8072)
