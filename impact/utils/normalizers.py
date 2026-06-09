from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import zarr

from impact.model.common.dict_of_tensor_mixin import DictOfTensorMixin
from impact.utils.pytorch_util import dict_apply, dict_apply_reduce, dict_apply_split


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        if "_default" not in self.params_dict:
            raise RuntimeError("Not initialized")
        params = self.params_dict["_default"]
        return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = {}
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            return dict_apply(input_stats, self.normalize)

        result = {}
        for key, group in input_stats.items():
            this_dict = {}
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            assert output_max > 0
            assert output_min < 0
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(
        stat,
        lambda x: {
            "pos": x[..., :3],
            "rot": x[..., 3:6],
            "gripper": x[..., 6:],
        },
    )

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        input_max = stat["max"]
        input_min = stat["min"]
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {"scale": scale, "offset": offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat["mean"])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    def get_gripper_param_info(stat):
        example = stat["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    pos_param, pos_info = get_pos_param_info(result["pos"])
    rot_param, rot_info = get_rot_param_info(result["rot"])
    gripper_param, gripper_info = get_gripper_param_info(result["gripper"])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], lambda x: np.concatenate(x, axis=-1)
    )
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], lambda x: np.concatenate(x, axis=-1)
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(stat, lambda x: {"pos": x[..., :3], "other": x[..., 3:]})

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        input_max = stat["max"]
        input_min = stat["min"]
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {"scale": scale, "offset": offset}, stat

    def get_other_param_info(stat):
        example = stat["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    pos_param, pos_info = get_pos_param_info(result["pos"])
    other_param, other_info = get_other_param_info(result["other"])

    param = dict_apply_reduce(
        [pos_param, other_param], lambda x: np.concatenate(x, axis=-1)
    )
    info = dict_apply_reduce(
        [pos_info, other_info], lambda x: np.concatenate(x, axis=-1)
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    da = stat["max"].shape[-1]
    dah = da // 2
    result = dict_apply_split(
        stat,
        lambda x: {
            "pos0": x[..., :3],
            "other0": x[..., 3:dah],
            "pos1": x[..., dah : dah + 3],
            "other1": x[..., dah + 3 :],
        },
    )

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        input_max = stat["max"]
        input_min = stat["min"]
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {"scale": scale, "offset": offset}, stat

    def get_other_param_info(stat):
        example = stat["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    pos0_param, pos0_info = get_pos_param_info(result["pos0"])
    pos1_param, pos1_info = get_pos_param_info(result["pos1"])
    other0_param, other0_info = get_other_param_info(result["other0"])
    other1_param, other1_info = get_other_param_info(result["other1"])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param],
        lambda x: np.concatenate(x, axis=-1),
    )
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info],
        lambda x: np.concatenate(x, axis=-1),
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    return stat


def _demo_normalizers():
    # Example shapes: obs = (B, T, D_obs), action = (B, T, D_act)
    obs = np.random.uniform(low=-2.0, high=2.0, size=(4, 16, 15)).astype(np.float32)
    action = np.random.uniform(low=-1.0, high=1.0, size=(4, 16, 8)).astype(np.float32)

    norm = LinearNormalizer()
    norm["proprioception"] = SingleFieldLinearNormalizer.create_fit(obs)
    norm["action"] = SingleFieldLinearNormalizer.create_fit(action)

    n_obs = norm["proprioception"].normalize(obs)
    n_act = norm["action"].normalize(action)
    obs_rec = norm["proprioception"].unnormalize(n_obs)
    act_rec = norm["action"].unnormalize(n_act)

    assert np.allclose(obs, obs_rec, atol=1e-5)
    assert np.allclose(action, act_rec, atol=1e-5)
    print("Normalizer demo passed: obs/action round-trip is consistent.")


if __name__ == "__main__":
    print("Test Normalizers")
    _demo_normalizers()
