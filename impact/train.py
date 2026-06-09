"""
Entrypoint for training diffusion policies on real-world datasets.
Usage:
    python -m impact.train
"""

import pathlib

import hydra
from omegaconf import OmegaConf

from impact.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")),
    config_name="default",
    version_base=None,
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
