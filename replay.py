import hydra
import pathlib
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy', 'config')),
    config_name="replay_cloud_data"
)
def main(cfg: OmegaConf):
    env_runner = hydra.utils.instantiate(cfg.task.env_runner)
    env_runner.run()

if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5679)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    main()