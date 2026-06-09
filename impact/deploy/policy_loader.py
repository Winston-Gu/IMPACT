from pathlib import Path

import dill
import hydra
import torch


def load_policy_from_checkpoint(
    checkpoint_path: str,
    device,
    prefer_ema: bool = True,
    num_inference_steps: int | None = None,
):
    path = Path(checkpoint_path).expanduser()
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    if num_inference_steps is not None and num_inference_steps > 0:
        cfg.policy.num_inference_steps = int(num_inference_steps)
    state_dicts = payload["state_dicts"]

    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if prefer_ema and "ema_model" in state_dicts else "model"
    if key not in state_dicts:
        raise RuntimeError(f"Checkpoint missing policy weights for '{key}'.")
    policy.load_state_dict(state_dicts[key])
    policy.to(device)
    policy.eval()
    return policy
