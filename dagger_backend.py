import hashlib
import time
import threading
import torch
import dill
import hydra
import os
import tempfile
import shutil
from omegaconf import OmegaConf, DictConfig
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import pathlib
from pathlib import Path
import logging
import asyncio
from collections import OrderedDict
import numpy as np
import base64
from datetime import datetime
import cv2

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached")

# ============================================================================
# DEBUG CONFIGURATION
# ============================================================================
DEBUG = True  # Set to True to enable debug output
DEBUG_DIR = Path(".cache/debug")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register OmegaConf resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# ============================================================================
# DEBUG HELPER FUNCTIONS
# ============================================================================
_debug_counter = 0

def save_obs_dict_for_debug(obs_dict: Dict[str, torch.Tensor], session_id: str = "unknown") -> None:
    """
    Save observation tensors to .cache directory for debugging.

    Args:
        obs_dict: Dictionary of observation tensors (already processed, on device)
                  Expected shape: {key: (batch, n_obs_steps, ...)}
        session_id: Optional session identifier for logging

    Saves:
        - Images as PNG files (.cache/debug/{timestamp}_{key}_frame{i}.png)
        - Low-dim data in text file (.cache/debug/{timestamp}_observations.txt)
    """
    if not DEBUG:
        return

    global _debug_counter

    try:
        # Create debug directory
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        # Generate unique timestamp and counter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _debug_counter += 1
        debug_prefix = f"{timestamp}_{_debug_counter:04d}"

        # Text file for metadata and low-dim data
        txt_path = DEBUG_DIR / f"{debug_prefix}_observations.txt"

        with open(txt_path, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Debug Counter: {_debug_counter}\n")
            f.write("="*80 + "\n\n")

            for key, tensor in obs_dict.items():
                f.write(f"[{key}]\n")
                f.write(f"  Shape: {tuple(tensor.shape)}\n")
                f.write(f"  Dtype: {tensor.dtype}\n")
                f.write(f"  Device: {tensor.device}\n")

                # Move tensor to CPU and convert to numpy
                tensor_cpu = tensor.detach().cpu().numpy()

                # Determine if this is an image or low-dim data
                # Image tensors typically have shape (batch, n_obs_steps, C, H, W)
                # Low-dim tensors have shape (batch, n_obs_steps, dim)

                if len(tensor_cpu.shape) == 5:  # (batch, n_obs_steps, C, H, W)
                    f.write(f"  Type: Image sequence\n")
                    batch_size, n_frames, C, H, W = tensor_cpu.shape
                    f.write(f"  Batch size: {batch_size}, Frames: {n_frames}, Channels: {C}, Height: {H}, Width: {W}\n")

                    # Save each frame for first batch item
                    for frame_idx in range(n_frames):
                        frame = tensor_cpu[0, frame_idx]  # (C, H, W)

                        # Convert CHW to HWC
                        frame_hwc = np.transpose(frame, (1, 2, 0)) * 255.0

                        # Convert to uint8 (assume values are in [0, 255] range)
                        frame_uint8 = np.clip(frame_hwc, 0, 255).astype(np.uint8)

                        # Convert RGB to BGR for cv2
                        if C == 3:
                            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame_uint8

                        # Save image
                        img_path = DEBUG_DIR / f"{debug_prefix}_{key}_frame{frame_idx}.png"
                        cv2.imwrite(str(img_path), frame_bgr)
                        f.write(f"  Frame {frame_idx}: saved to {img_path.name}\n")
                        f.write(f"    Min: {frame.min():.3f}, Max: {frame.max():.3f}, Mean: {frame.mean():.3f}\n")

                elif len(tensor_cpu.shape) == 3:  # (batch, n_obs_steps, dim)
                    f.write(f"  Type: Low-dimensional sequence\n")
                    batch_size, n_frames, dim = tensor_cpu.shape
                    f.write(f"  Batch size: {batch_size}, Frames: {n_frames}, Dimension: {dim}\n")

                    # Write all frames for first batch item
                    for frame_idx in range(n_frames):
                        frame_data = tensor_cpu[0, frame_idx]
                        f.write(f"  Frame {frame_idx}: {np.array2string(frame_data, precision=4, separator=', ')}\n")

                else:
                    f.write(f"  Type: Unknown (unexpected shape)\n")
                    f.write(f"  Data (first batch item): {np.array2string(tensor_cpu[0], precision=4, separator=', ', threshold=100)}\n")

                f.write("\n")

        logger.info(f"[DEBUG] Saved observations to {txt_path}")

    except Exception as e:
        logger.error(f"[DEBUG] Failed to save observations: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

class InferenceRequest(BaseModel):
    request_id: str
    device_id: str
    workspace_config: str
    task_config: str
    checkpoint_path: str
    observations: Dict[str, Any]

class InferenceResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class InferenceSession:
    def __init__(self, session_id: str, workspace_config: str, task_config: str, checkpoint_path: str):
        self.session_id = session_id
        self.workspace_config = workspace_config
        self.task_config = task_config
        self.initial_checkpoint_path = checkpoint_path  # Immutable, used for session identification
        self.checkpoint_path = checkpoint_path  # Mutable, can be updated via hot-reload
        self.last_access = time.time()
        self.workspace: Optional[BaseWorkspace] = None
        self.policy: Optional[BaseImagePolicy] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.initialized = False
        self.n_obs_steps = None
        # Reference counting for thread-safe cleanup
        self._ref_count = 0
        self._ref_lock = threading.Lock()
    
    def acquire(self):
        """Increment reference count to prevent cleanup while in use."""
        with self._ref_lock:
            self._ref_count += 1
    
    def release(self):
        """Decrement reference count."""
        with self._ref_lock:
            self._ref_count -= 1
    
    def is_in_use(self) -> bool:
        """Check if session is currently being used."""
        with self._ref_lock:
            return self._ref_count > 0
        
    def _load_config(self) -> DictConfig:
        """Load and merge configurations"""
        config_path = Path("diffusion_policy/config")
        
        # Load workspace config
        workspace_config_path = config_path / f"{self.workspace_config}.yaml"
        if not workspace_config_path.exists():
            raise FileNotFoundError(f"Workspace config not found: {workspace_config_path}")
        
        # Use Hydra to load config with proper task override
        with hydra.initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
            cfg = hydra.compose(
                config_name=self.workspace_config,
                overrides=[f"task={self.task_config}"]
            )
        
        return cfg
    
    def _initialize_session(self):
        """Initialize workspace and policy for this session"""
        try:
            logger.info(f"Initializing session {self.session_id}")
            
            # Load configuration
            cfg = self._load_config()
            
            # Load checkpoint
            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
            payload = torch.load(open(self.checkpoint_path, 'rb'), pickle_module=dill)
            
            # Create workspace
            cls = hydra.utils.get_class(cfg._target_)
            self.workspace = cls(cfg)
            self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            
            # Get policy
            if 'diffusion' in cfg.name:
                self.policy: BaseImagePolicy = self.workspace.model
                if cfg.training.use_ema:
                    self.policy = self.workspace.ema_model

                self.policy.eval().to(self.device)

                # Set inference parameters
                self.policy.num_inference_steps = 8
                # Get n_obs_steps from config and store as instance variable
                self.n_obs_steps = cfg.n_obs_steps
                horizon = cfg.horizon
                self.policy.n_action_steps = horizon - self.n_obs_steps + 1
            else:
                raise NotImplementedError("Non-diffusion models not supported")
            
            # Warm up with dummy inference
            self._warmup()
            
            self.initialized = True
            logger.info(f"Session {self.session_id} initialized successfully")
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize session {self.session_id}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self._cleanup_internal()  # Use internal cleanup since we're already holding the lock
            raise
    
    def _warmup(self):
        """Perform a dummy inference to warm up the model"""
        try:
            # Create dummy observations based on shape_meta
            shape_meta = self.workspace.cfg.shape_meta.obs
            dummy_obs = {}
            
            for key, spec in shape_meta.items():
                shape = spec.shape
                if spec.type == 'rgb':
                    # Create dummy RGB image
                    dummy_obs[key] = torch.randn(1, self.n_obs_steps, *shape).to(self.device)
                elif spec.type == 'low_dim':
                    # Create dummy low-dimensional data
                    dummy_obs[key] = torch.randn(1, self.n_obs_steps, *shape).to(self.device)
            
            # Perform dummy inference
            with torch.no_grad():
                _ = self.policy.predict_action(dummy_obs)
            
            logger.info(f"Session {self.session_id} warmed up successfully")
        except Exception as e:
            logger.warning(f"Warmup failed for session {self.session_id}: {e}")

    def predict_action(self, observations: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Perform inference with the loaded policy"""
        with self.lock:
            if not self.initialized:
                self._initialize_session()

            self.last_access = time.time()
            logger.info(f"[Inference] request_id={request_id}, checkpoint={self.checkpoint_path}")

            try:
                # Convert observations to torch tensors
                obs_dict = {}
                for key, value in observations.items():
                    if isinstance(value, list) and isinstance(value[0], str):  # multi-frame base64 encoded images
                        frames = []
                        for frame_b64 in value:
                            decoded = base64.b64decode(frame_b64)
                            img_array = np.frombuffer(decoded, dtype=np.uint8)

                            # Get shape from config
                            shape_meta = self.workspace.cfg.shape_meta.obs[key]
                            config_shape = [int(s) for s in shape_meta.shape]
                            if len(config_shape) == 3:  # [C, H, W]
                                single_frame_shape = config_shape
                            elif len(config_shape) == 4:  # [n_obs_steps, C, H, W]
                                single_frame_shape = config_shape[1:]
                            else:
                                # Fallback: assume CHW for images
                                single_frame_shape = [3, 240, 320]

                            img_array = img_array.reshape(*single_frame_shape)
                            frames.append(img_array)

                        # Stack frames: (n_obs_steps, C, H, W)
                        multi_frame_array = np.stack(frames, axis=0)
                        tensor = torch.from_numpy(multi_frame_array).float() / 255.0
                        tensor = tensor.unsqueeze(0)  # Add batch dimension: (1, n_obs_steps, C, H, W)
                        obs_dict[key] = tensor.to(self.device)

                    elif isinstance(value, list) and isinstance(value[0], list):  # multi-frame low-dim data
                        # Stack frames: (n_obs_steps, dim)
                        multi_frame_array = np.array(value)
                        tensor = torch.from_numpy(multi_frame_array).float()
                        tensor = tensor.unsqueeze(0)  # Add batch dimension: (1, n_obs_steps, dim)
                        obs_dict[key] = tensor.to(self.device)

                    elif isinstance(value, str):  # single base64 encoded image (fallback)
                        decoded = base64.b64decode(value)
                        img_array = np.frombuffer(decoded, dtype=np.uint8)
                        shape_meta = self.workspace.cfg.shape_meta.obs[key]
                        shape_ints = [int(s) for s in shape_meta.shape]
                        img_array = img_array.reshape(*shape_ints)
                        tensor = torch.from_numpy(img_array).float() / 255.0
                        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
                        obs_dict[key] = tensor.to(self.device)

                    else:  # single low-dim data (fallback)
                        tensor = torch.from_numpy(np.array(value)).float()
                        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
                        obs_dict[key] = tensor.to(self.device)

                # Debug: save obs_dict to .cache directory
                save_obs_dict_for_debug(obs_dict, session_id=self.session_id)

                # Perform inference
                with torch.no_grad():
                    action_dict = self.policy.predict_action(obs_dict)
                
                # Convert back to numpy
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
                
                return {
                    "action": np_action_dict["action"].tolist(),
                    "session_id": self.session_id,
                    "request_id": request_id
                }
                
            except Exception as e:
                import traceback
                logger.error(f"Inference failed for session {self.session_id}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise
    
    def _cleanup_internal(self):
        """Internal cleanup without lock acquisition. Must be called with self.lock held."""
        logger.info(f"Cleaning up session {self.session_id}")
        if self.policy is not None:
            del self.policy
            self.policy = None
        if self.workspace is not None:
            del self.workspace
            self.workspace = None
        torch.cuda.empty_cache()
        self.initialized = False
    
    def cleanup(self, wait_timeout: float = 5.0):
        """
        Clean up resources. Thread-safe: waits for ongoing operations to complete.
        
        Args:
            wait_timeout: Maximum time to wait for ongoing operations (seconds)
        """
        # Wait for ongoing operations to complete
        start_time = time.time()
        while self.is_in_use():
            if time.time() - start_time > wait_timeout:
                logger.warning(f"Session {self.session_id} cleanup timeout, forcing cleanup")
                break
            time.sleep(0.01)
        
        with self.lock:
            self._cleanup_internal()
    
    def reload_checkpoint(self, new_checkpoint_path: str) -> bool:
        """
        Hot-reload checkpoint without full re-initialization.
        Thread-safe: acquires lock before reloading.
        
        Args:
            new_checkpoint_path: Path to the new checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                logger.info(f"Reloading checkpoint for session {self.session_id}")
                logger.info(f"  Old checkpoint: {self.checkpoint_path}")
                logger.info(f"  New checkpoint: {new_checkpoint_path}")
                
                if not Path(new_checkpoint_path).exists():
                    logger.error(f"New checkpoint not found: {new_checkpoint_path}")
                    return False
                
                # Load new checkpoint payload
                payload = torch.load(open(new_checkpoint_path, 'rb'), pickle_module=dill)
                
                # Reload state dict into existing workspace/policy
                if self.workspace is not None:
                    self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)
                    
                    # Re-extract policy
                    cfg = self.workspace.cfg
                    if 'diffusion' in cfg.name:
                        self.policy = self.workspace.model
                        if cfg.training.use_ema:
                            self.policy = self.workspace.ema_model
                        self.policy.eval().to(self.device)
                
                # Update checkpoint path
                old_checkpoint_path = self.checkpoint_path
                self.checkpoint_path = new_checkpoint_path
                self.last_access = time.time()
                
                # Warmup with new weights
                self._warmup()
                
                logger.info(f"Session {self.session_id} checkpoint reloaded successfully")
                return True
                
            except Exception as e:
                import traceback
                logger.error(f"Failed to reload checkpoint for session {self.session_id}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                return False


class CheckpointUpdateManager:
    """
    Manages checkpoint updates from training server.
    Provides thread-safe checkpoint reception and session updates.
    """
    
    def __init__(self, checkpoint_dir: str = ".cache/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._update_in_progress = False
        self._current_checkpoint: Optional[str] = None
        self._update_history = []
    
    def receive_checkpoint(
        self,
        checkpoint_file: bytes,
        workspace_config: str,
        task_config: str,
        sha256: str,
        file_size: int,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Receive and validate a checkpoint file.
        
        Args:
            checkpoint_file: Checkpoint file bytes
            workspace_config: Workspace configuration name
            task_config: Task configuration name  
            sha256: Expected SHA256 hash
            file_size: Expected file size
            metadata: Additional metadata (global_step, epoch, etc.)
            
        Returns:
            Path to saved checkpoint or None if failed
        """
        with self._lock:
            try:
                self._update_in_progress = True
                
                # Validate file size
                if len(checkpoint_file) != file_size:
                    logger.error(f"Checkpoint size mismatch: {len(checkpoint_file)} != {file_size}")
                    return None
                
                # Validate SHA256
                computed_sha256 = hashlib.sha256(checkpoint_file).hexdigest()
                if computed_sha256 != sha256:
                    logger.error(f"Checkpoint SHA256 mismatch: {computed_sha256} != {sha256}")
                    return None
                
                # Generate checkpoint filename
                global_step = metadata.get('global_step', 0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"{workspace_config}_{task_config}_step{global_step}_{timestamp}.ckpt"
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                
                # Write checkpoint file
                with open(checkpoint_path, 'wb') as f:
                    f.write(checkpoint_file)
                
                logger.info(f"Checkpoint saved to: {checkpoint_path}")
                
                # Update state
                self._current_checkpoint = str(checkpoint_path)
                self._update_history.append({
                    'path': str(checkpoint_path),
                    'timestamp': timestamp,
                    'metadata': metadata,
                })
                
                # Keep only last 5 checkpoints
                self._cleanup_old_checkpoints()
                
                return str(checkpoint_path)
                
            except Exception as e:
                logger.error(f"Failed to receive checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            finally:
                self._update_in_progress = False
    
    def _cleanup_old_checkpoints(self, keep_n: int = 5):
        """Remove old checkpoints, keeping only the most recent N."""
        try:
            if len(self._update_history) > keep_n:
                old_entries = self._update_history[:-keep_n]
                self._update_history = self._update_history[-keep_n:]
                
                for entry in old_entries:
                    old_path = Path(entry['path'])
                    if old_path.exists():
                        old_path.unlink()
                        logger.info(f"Removed old checkpoint: {old_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    @property
    def current_checkpoint(self) -> Optional[str]:
        with self._lock:
            return self._current_checkpoint
    
    @property
    def is_updating(self) -> bool:
        with self._lock:
            return self._update_in_progress


class SessionManager:
    def __init__(self, max_sessions: int = 2, session_timeout: int = 60):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.sessions: OrderedDict[str, InferenceSession] = OrderedDict()
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_session_id(self, workspace_config: str, task_config: str, checkpoint_path: str) -> str:
        """Generate a unique session ID based on configuration"""
        config_str = f"{workspace_config}#{task_config}#{checkpoint_path}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _evict_oldest_session(self):
        """Evict the oldest session to make room for new one"""
        if self.sessions:
            oldest_session_id = next(iter(self.sessions))
            oldest_session = self.sessions.pop(oldest_session_id)
            oldest_session.cleanup()
            logger.info(f"Evicted oldest session: {oldest_session_id}")
    
    def get_or_create_session(self, workspace_config: str, task_config: str, checkpoint_path: str) -> InferenceSession:
        """Get existing session or create new one"""
        session_id = self._generate_session_id(workspace_config, task_config, checkpoint_path)

        with self.lock:
            # Check if session already exists
            if session_id in self.sessions:
                # Move to end (most recently used)
                session = self.sessions.pop(session_id)
                self.sessions[session_id] = session
                return session

            # Check if we need to evict sessions
            while len(self.sessions) >= self.max_sessions:
                self._evict_oldest_session()

            # Create new session
            session = InferenceSession(session_id, workspace_config, task_config, checkpoint_path)
            self.sessions[session_id] = session

            logger.info(f"Created new session: {session_id}")
            return session
    
    def _cleanup_loop(self):
        """Background thread to clean up expired sessions"""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                with self.lock:
                    for session_id, session in list(self.sessions.items()):
                        if current_time - session.last_access > self.session_timeout:
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        session = self.sessions.pop(session_id)
                        session.cleanup()
                        logger.info(f"Cleaned up expired session: {session_id}")
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(10)
    
    def update_all_sessions_checkpoint(
        self,
        workspace_config: str,
        task_config: str,
        new_checkpoint_path: str
    ) -> Dict[str, bool]:
        """
        Update checkpoint for all matching sessions.
        
        Session ID is based on initial_checkpoint_path and remains unchanged
        after checkpoint hot-reload, allowing session reuse.
        
        Args:
            workspace_config: Target workspace config
            task_config: Target task config
            new_checkpoint_path: Path to new checkpoint
            
        Returns:
            Dict mapping session_id to update success status
        """
        results = {}
        
        with self.lock:
            print(self.sessions)
            for session_id, session in list(self.sessions.items()):
                if (session.workspace_config == workspace_config and 
                    session.task_config == task_config):
                    success = session.reload_checkpoint(new_checkpoint_path)
                    results[session_id] = success
                    # Session ID remains unchanged since it's based on initial_checkpoint_path
        
        return results
    
    def get_sessions_info(self) -> Dict[str, Dict]:
        """Get information about all active sessions."""
        with self.lock:
            info = {}
            for session_id, session in self.sessions.items():
                info[session_id] = {
                    'workspace_config': session.workspace_config,
                    'task_config': session.task_config,
                    'initial_checkpoint_path': session.initial_checkpoint_path,
                    'checkpoint_path': session.checkpoint_path,
                    'initialized': session.initialized,
                    'last_access': session.last_access,
                }
            return info


# Global managers
session_manager = SessionManager(max_sessions=2, session_timeout=60)
checkpoint_update_manager = CheckpointUpdateManager()

# FastAPI app
app = FastAPI(title="UMI Backend Inference Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "umi-backend-inference"}

@app.post("/backend/inference", response_model=InferenceResponse)
async def backend_inference(request: InferenceRequest):
    session = None
    try:
        logger.info(f"Received inference request for device {request.device_id}, request_id: {request.request_id}")

        # Get or create session
        session = session_manager.get_or_create_session(
            request.workspace_config,
            request.task_config,
            request.checkpoint_path
        )
        
        # Acquire reference to prevent cleanup during inference
        session.acquire()
        
        # Perform inference
        result = session.predict_action(request.observations, request.request_id)
        
        return InferenceResponse(success=True, data=result)
        
    except Exception as e:
        logger.error(f"Backend inference failed: {e}")
        return InferenceResponse(success=False, error=str(e))
    finally:
        if session is not None:
            session.release()


class CheckpointUpdateResponse(BaseModel):
    success: bool
    checkpoint_path: Optional[str] = None
    updated_sessions: Optional[Dict[str, bool]] = None
    error: Optional[str] = None


@app.post("/backend/update_checkpoint", response_model=CheckpointUpdateResponse)
async def update_checkpoint(
    checkpoint_file: UploadFile = File(...),
    workspace_config: str = Form(...),
    task_config: str = Form(...),
    sha256: str = Form(...),
    file_size: str = Form(...),
    global_step: str = Form("0"),
    epoch: str = Form("0"),
):
    """
    Receive a checkpoint file from the training server and update all matching sessions.
    
    This endpoint:
    1. Validates and saves the checkpoint file
    2. Updates all active sessions with matching workspace/task config
    3. Returns update status for each affected session
    """
    try:
        logger.info(f"Received checkpoint update request")
        logger.info(f"  workspace_config: {workspace_config}")
        logger.info(f"  task_config: {task_config}")
        logger.info(f"  file_size: {file_size}")
        logger.info(f"  global_step: {global_step}")
        
        # Read checkpoint file
        checkpoint_bytes = await checkpoint_file.read()
        
        # Receive and validate checkpoint
        metadata = {
            'global_step': int(global_step),
            'epoch': int(epoch),
        }
        
        checkpoint_path = checkpoint_update_manager.receive_checkpoint(
            checkpoint_file=checkpoint_bytes,
            workspace_config=workspace_config,
            task_config=task_config,
            sha256=sha256,
            file_size=int(file_size),
            metadata=metadata,
        )
        
        if checkpoint_path is None:
            return CheckpointUpdateResponse(
                success=False,
                error="Failed to receive or validate checkpoint"
            )
        
        # Update all matching sessions
        updated_sessions = session_manager.update_all_sessions_checkpoint(
            workspace_config=workspace_config,
            task_config=task_config,
            new_checkpoint_path=checkpoint_path,
        )
        
        logger.info(f"Checkpoint update complete. Updated sessions: {updated_sessions}")
        
        return CheckpointUpdateResponse(
            success=True,
            checkpoint_path=checkpoint_path,
            updated_sessions=updated_sessions,
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Checkpoint update failed: {e}")
        logger.error(traceback.format_exc())
        return CheckpointUpdateResponse(
            success=False,
            error=str(e)
        )


@app.get("/backend/sessions")
async def get_sessions():
    """Get information about all active inference sessions."""
    return {
        "sessions": session_manager.get_sessions_info(),
        "current_checkpoint": checkpoint_update_manager.current_checkpoint,
        "update_in_progress": checkpoint_update_manager.is_updating,
    }


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(4071)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
