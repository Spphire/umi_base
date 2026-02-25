import os
import pathlib
import dill
import torch
import hydra
from omegaconf import OmegaConf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np

# Avoid OpenMP duplicate load issue (Windows common, harmless on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = FastAPI(title="UMI Simple Inference Server", version="1.0.0")

# 启动时加载模型
MODEL_CKPT = os.environ.get("MODEL_CKPT", "./latest.ckpt")
MODEL_CFG = os.environ.get("MODEL_CFG", None)

def load_policy(ckpt_path: str, cfg_yaml_path: str = None):
    ckpt_path = pathlib.Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    if cfg_yaml_path is not None:
        cfg_yaml_path = pathlib.Path(cfg_yaml_path).expanduser().resolve()
        if not cfg_yaml_path.is_file():
            raise FileNotFoundError(f"Config yaml not found: {cfg_yaml_path}")
        yaml_cfg = OmegaConf.load(str(cfg_yaml_path))
        base_cfg = OmegaConf.create(payload["cfg"])
        OmegaConf.set_struct(base_cfg, False)
        cfg = OmegaConf.merge(base_cfg, yaml_cfg)
    else:
        cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=['lr_scheduler'], include_keys=None)
    policy = workspace.model
    if hasattr(cfg, "training") and getattr(cfg.training, "use_ema", False) and hasattr(workspace, "ema_model"):
        policy = workspace.ema_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    return policy, cfg, device

policy, cfg, device = load_policy(MODEL_CKPT, MODEL_CFG)

@app.post("/v1/inference")
async def inference_endpoint(request: Request):
    try:
        data = await request.json()
        obs = {}
        # 预处理 base64 图片和 wrist/eye
        for k, v in data.items():
            if k.endswith("_image") and isinstance(v, str):
                import base64
                import io
                from PIL import Image, ImageOps
                img_bytes = base64.b64decode(v)
                if "wrist" in k:
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    if arr.size == 3*224*224:
                        arr = arr.reshape((3, 224, 224))
                        obs[k] = arr
                    else:
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
                    obs[k] = arr
                else:
                    obs[k] = v
            else:
                obs[k] = v
        # 转 tensor
        obs_tensor = {kk: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, (list, tuple, np.ndarray)) else v for kk, v in obs.items()}
        with torch.no_grad():
            result = policy.predict_action(obs_tensor)
        out = {k: v[0].detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in result.items()}
        return JSONResponse(content={"success": True, "data": out})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "umi-simple-inference-server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8072)
