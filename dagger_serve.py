import uvicorn
import base64
import requests
import yaml
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached")

app = FastAPI(title="UMI Inference Server", version="1.0.0")

class InferenceRequest(BaseModel):
    request_id: str = Field(..., description="Request ID from client (UUID string)")
    device_id: str = Field(..., description="iPhone device ID")
    workspace_config: str = Field(..., description="Workspace configuration name (e.g., train_diffusion_unet_timm_workspace)")
    task_config: str = Field(..., description="Task configuration name (e.g., real_pick_and_place_image_iphone)")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    observations: Dict[str, Any] = Field(..., description="Observation data matching task requirements")
    debug: bool = Field(default=False, description="Enable debug mode to save observations to .cache directory")

class InferenceResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TaskObsSpec:
    def __init__(self, task_config_name: str, workspace_config_name: str):
        self.task_config_name = task_config_name
        self.workspace_config_name = workspace_config_name
        self.obs_spec = self._load_obs_spec()

    def _load_obs_spec(self) -> Dict[str, Dict[str, Any]]:
        try:
            # Use Hydra to load complete config with proper composition
            config_path = Path("diffusion_policy/config")

            # Register OmegaConf resolver
            OmegaConf.register_new_resolver("eval", eval, replace=True)
            with hydra.initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
                cfg = hydra.compose(
                    config_name=self.workspace_config_name,
                    overrides=[f"task={self.task_config_name}"]
                )

            # Extract shape_meta from loaded config
            n_obs_steps = cfg.n_obs_steps
            shape_meta = cfg.get('shape_meta', {})
            obs_spec = {}

            if 'obs' in shape_meta:
                for key, spec in shape_meta.obs.items():
                    obs_type = spec.type
                    shape = spec.shape

                    # Convert OmegaConf lists to regular lists
                    if hasattr(shape, '__iter__'):
                        shape = [int(s) for s in shape]
                    else:
                        shape = [int(shape)]

                    # For iPhone inference, we always send n_obs_steps frames
                    # Determine if this is rgb or low_dim based on type
                    if obs_type == 'rgb':
                        # shape from config is [C, H, W], we need [n_obs_steps, C, H, W]
                        obs_spec[key] = {
                            "shape": [n_obs_steps] + shape,
                            "type": "rgb_sequence"
                        }
                    elif obs_type == 'low_dim':
                        # shape from config is [dim], we need [n_obs_steps, dim]
                        obs_spec[key] = {
                            "shape": [n_obs_steps] + shape,
                            "type": "low_dim_sequence"
                        }
                    else:
                        raise ValueError(f"Unknown observation type: {obs_type}")

            return obs_spec

        except Exception as e:
            import traceback
            raise ValueError(f"Failed to load config {self.workspace_config_name} with task {self.task_config_name}: {e}\n{traceback.format_exc()}")
    
    def validate_observations(self, observations: Dict[str, Any]) -> tuple[bool, str]:
        required_keys = set(self.obs_spec.keys())
        provided_keys = set(observations.keys())
        
        missing_keys = required_keys - provided_keys
        if missing_keys:
            return False, f"Missing required observation keys: {missing_keys}"
        
        extra_keys = provided_keys - required_keys
        if extra_keys:
            return False, f"Unexpected observation keys: {extra_keys}"
        
        for key, spec in self.obs_spec.items():
            obs_data = observations[key]
            expected_shape = spec['shape']
            obs_type = spec['type']
            
            if obs_type == 'rgb':
                if not isinstance(obs_data, str):
                    return False, f"Observation {key} should be base64 encoded string for rgb type"
                try:
                    decoded = base64.b64decode(obs_data)
                    img_array = np.frombuffer(decoded, dtype=np.uint8)
                    # Convert expected_shape to integers and calculate size
                    shape_ints = [int(s) for s in expected_shape]
                    expected_size = np.prod(shape_ints)
                    if len(img_array) < expected_size:
                        return False, f"Observation {key} decoded size {len(img_array)} < expected {expected_size}"
                except Exception as e:
                    return False, f"Failed to decode base64 image for {key}: {e}"
            
            elif obs_type == 'rgb_sequence':
                if not isinstance(obs_data, list):
                    return False, f"Observation {key} should be list of base64 strings for rgb_sequence type"
                
                n_frames, c, h, w = expected_shape
                if len(obs_data) != n_frames:
                    return False, f"Observation {key} has {len(obs_data)} frames, expected {n_frames}"
                
                for i, frame_data in enumerate(obs_data):
                    if not isinstance(frame_data, str):
                        return False, f"Observation {key} frame {i} should be base64 encoded string"
                    try:
                        decoded = base64.b64decode(frame_data)
                        img_array = np.frombuffer(decoded, dtype=np.uint8)
                        expected_frame_size = c * h * w
                        if len(img_array) != expected_frame_size:
                            return False, f"Observation {key} frame {i} size {len(img_array)} != expected {expected_frame_size}"
                    except Exception as e:
                        return False, f"Failed to decode base64 image for {key} frame {i}: {e}"
            
            elif obs_type == 'low_dim':
                if not isinstance(obs_data, (list, tuple)):
                    return False, f"Observation {key} should be list/array for low_dim type"
                if len(obs_data) != expected_shape[0]:
                    return False, f"Observation {key} has shape {len(obs_data)}, expected {expected_shape[0]}"
            
            elif obs_type == 'low_dim_sequence':
                if not isinstance(obs_data, list):
                    return False, f"Observation {key} should be list for low_dim_sequence type"
                
                n_frames, dim = expected_shape
                if len(obs_data) != n_frames:
                    return False, f"Observation {key} has {len(obs_data)} frames, expected {n_frames}"
                
                for i, frame_data in enumerate(obs_data):
                    if not isinstance(frame_data, (list, tuple)):
                        return False, f"Observation {key} frame {i} should be list/array"
                    if len(frame_data) != dim:
                        return False, f"Observation {key} frame {i} has dim {len(frame_data)}, expected {dim}"
        
        return True, "Valid"

def get_task_config_path(task_name: str) -> str:
    config_dir = Path("diffusion_policy/config/task")
    task_file = config_dir / f"{task_name}.yaml"
    if not task_file.exists():
        raise ValueError(f"Task configuration {task_name} not found at {task_file}")
    return str(task_file)

def get_workspace_config_path(workspace_name: str) -> str:
    config_dir = Path("diffusion_policy/config")
    workspace_file = config_dir / f"{workspace_name}.yaml"
    if not workspace_file.exists():
        raise ValueError(f"Workspace configuration {workspace_name} not found at {workspace_file}")
    return str(workspace_file)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "umi-inference-server"}

@app.post("/v1/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    try:
        # Validate task configuration exists
        get_task_config_path(request.task_config)

        # Validate workspace configuration exists
        get_workspace_config_path(request.workspace_config)

        # Validate checkpoint exists
        if not os.path.exists(request.checkpoint_path):
            print(f"Checkpoint not found: {request.checkpoint_path}")
            raise HTTPException(status_code=400, detail=f"Checkpoint not found: {request.checkpoint_path}")

        # Validate observation format by loading actual config
        # Assume n_obs_steps=2 for iPhone inference (standard multi-frame setup)
        task_spec = TaskObsSpec(request.task_config, request.workspace_config)
        is_valid, error_msg = task_spec.validate_observations(request.observations)
        
        if not is_valid:
            print(f"Invalid observation format: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Invalid observation format: {error_msg}")
        
        # Forward request to internal backend API
        backend_payload = {
            "request_id": request.request_id,
            "device_id": request.device_id,
            "workspace_config": request.workspace_config,
            "task_config": request.task_config,
            "checkpoint_path": request.checkpoint_path,
            "observations": request.observations,
            "debug": request.debug
        }
        
        try:
            # TODO: Update this URL to match your actual backend service
            backend_url = "http://localhost:8080/backend/inference"
            response = requests.post(
                backend_url, 
                json=backend_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return InferenceResponse(
                    success=True, 
                    data=response.json()
                )
            else:
                return InferenceResponse(
                    success=False,
                    error=f"Backend service error: {response.status_code} - {response.text}"
                )
                
        except requests.exceptions.RequestException as e:
            return InferenceResponse(
                success=False,
                error=f"Failed to connect to backend service: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        return InferenceResponse(
            success=False,
            error=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8072)
