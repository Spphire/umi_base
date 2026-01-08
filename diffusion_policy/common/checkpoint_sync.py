import os
import hashlib
import requests
import time
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class CheckpointSyncServer:
    """
    HTTP server for pushing checkpoints from training server (Machine A) 
    to inference server (Machine B).
    
    Supports:
    - File transfer with chunked upload
    - SHA256 verification
    - Retry mechanism
    - Timeout handling
    """
    
    def __init__(
        self,
        inference_server_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks
    ):
        self.inference_server_url = inference_server_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.chunk_size = chunk_size
        
        self._session = requests.Session()
    
    def _compute_sha256(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def push_checkpoint(
        self,
        checkpoint_path: str,
        workspace_config: str,
        task_config: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Push a checkpoint file to the inference server.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            workspace_config: Workspace configuration name
            task_config: Task configuration name
            metadata: Optional metadata dict (e.g., training step, epoch)
        
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
        
        file_size = checkpoint_path.stat().st_size
        sha256 = self._compute_sha256(str(checkpoint_path))
        
        logger.info(f"Pushing checkpoint: {checkpoint_path}")
        logger.info(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        logger.info(f"  SHA256: {sha256}")
        
        for attempt in range(self.max_retries):
            try:
                success = self._upload_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    workspace_config=workspace_config,
                    task_config=task_config,
                    sha256=sha256,
                    file_size=file_size,
                    metadata=metadata or {},
                )
                if success:
                    logger.info(f"Checkpoint pushed successfully to {self.inference_server_url}")
                    return True
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"Failed to push checkpoint after {self.max_retries} attempts")
        return False
    
    def _upload_checkpoint(
        self,
        checkpoint_path: str,
        workspace_config: str,
        task_config: str,
        sha256: str,
        file_size: int,
        metadata: Dict[str, Any],
    ) -> bool:
        url = f"{self.inference_server_url}/backend/update_checkpoint"
        
        with open(checkpoint_path, 'rb') as f:
            files = {
                'checkpoint_file': (os.path.basename(checkpoint_path), f, 'application/octet-stream')
            }
            data = {
                'workspace_config': workspace_config,
                'task_config': task_config,
                'sha256': sha256,
                'file_size': str(file_size),
                'global_step': str(metadata.get('global_step', 0)),
                'epoch': str(metadata.get('epoch', 0)),
            }
            
            response = self._session.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout,
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                return True
            else:
                logger.error(f"Server returned error: {result.get('error', 'Unknown error')}")
                return False
        else:
            logger.error(f"Server returned status {response.status_code}: {response.text}")
            return False
    
    def notify_update(
        self,
        workspace_config: str,
        task_config: str,
        checkpoint_url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Notify inference server to pull checkpoint from a URL.
        Alternative to push_checkpoint for large files or shared storage.
        
        Args:
            workspace_config: Workspace configuration name
            task_config: Task configuration name
            checkpoint_url: URL where the checkpoint can be downloaded
            metadata: Optional metadata dict
        
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.inference_server_url}/backend/notify_checkpoint_update"
        
        payload = {
            'workspace_config': workspace_config,
            'task_config': task_config,
            'checkpoint_url': checkpoint_url,
            'metadata': metadata or {},
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success', False):
                        logger.info(f"Checkpoint update notification sent successfully")
                        return True
                    else:
                        logger.error(f"Server returned error: {result.get('error', 'Unknown error')}")
                else:
                    logger.error(f"Server returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"Failed to notify checkpoint update after {self.max_retries} attempts")
        return False
    
    def check_server_health(self) -> bool:
        """Check if the inference server is healthy and reachable."""
        try:
            response = self._session.get(
                f"{self.inference_server_url}/health",
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def close(self):
        """Close the HTTP session."""
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
