from typing import Dict

from impact.policy.base_image_policy import BaseImagePolicy
from impact.runner.base_image_runner import BaseImageRunner


class OfflineImageRunner(BaseImageRunner):
    """Lightweight runner placeholder for real-world datasets."""

    def __init__(self, output_dir: str = None):
        super().__init__(output_dir=output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        # No simulator available; return an empty dict for logging compatibility.
        policy.eval()
        return {}
