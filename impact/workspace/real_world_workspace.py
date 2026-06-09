import copy
import os
import random
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from impact.dataset.base_dataset import BaseImageDataset
from impact.model.common.lr_scheduler import get_scheduler
from impact.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from impact.runner.base_image_runner import BaseImageRunner
from impact.utils.json_logger import JsonLogger
from impact.utils.pytorch_util import dict_apply, optimizer_to
from impact.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RealWorldWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: Optional[DiffusionUnetImagePolicy] = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0
        self._wandb_run = None
        self._tb_writer: Optional[SummaryWriter] = None

    def _init_loggers(self):
        cfg = self.cfg
        log_cfg = cfg.logging
        exp_name = cfg.get("experiment_name", "exp")
        output_dir = self.output_dir

        wandb_run = None
        if log_cfg.use_wandb:
            wandb_run = wandb.init(
                dir=str(output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **log_cfg.wandb,
            )
            wandb_run.name = exp_name
            wandb_run.config.update({"output_dir": output_dir})
        self._wandb_run = wandb_run

        if log_cfg.enable_tensorboard:
            self._tb_writer = SummaryWriter(log_cfg.tensorboard_dir)

    def _log(self, step_log: dict, step: int):
        if self._wandb_run is not None:
            self._wandb_run.log(step_log, step=step)
        if self._tb_writer is not None:
            for key, value in step_log.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, step)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt = self.get_checkpoint_path()
            if latest_ckpt.is_file():
                self.load_checkpoint(path=latest_ckpt)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.dataset)
        train_loader = DataLoader(dataset, **cfg.dataloader.train_data_loader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_loader = DataLoader(val_dataset, **cfg.dataloader.val_data_loader)

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_loader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner: Optional[BaseImageRunner] = None
        if cfg.runner is not None:
            env_runner = hydra.utils.instantiate(cfg.runner, output_dir=self.output_dir)

        self._init_loggers()

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = []
                with tqdm.tqdm(
                    train_loader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if ema is not None:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self._log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                train_loss = (
                    float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
                )
                step_log["train_loss"] = train_loss

                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                if (
                    env_runner is not None
                    and (self.epoch % cfg.training.rollout_every) == 0
                ):
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                if (self.epoch % cfg.training.val_every) == 0:
                    model_was_training = self.model.training
                    ema_was_training = (
                        None if self.ema_model is None else self.ema_model.training
                    )
                    self.model.eval()
                    if self.ema_model is not None:
                        self.ema_model.eval()
                    try:
                        with torch.no_grad():
                            val_losses = []
                            with tqdm.tqdm(
                                val_loader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(
                                        batch, lambda x: x.to(device, non_blocking=True)
                                    )
                                    loss = self.model.compute_loss(batch)
                                    # val_losses.append(loss)
                                    val_losses.append(loss.detach().float().cpu())
                                    if (
                                        cfg.training.max_val_steps is not None
                                    ) and batch_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            if len(val_losses) > 0:
                                # val_loss = torch.mean(torch.tensor(val_losses)).item()
                                val_loss = torch.stack(val_losses).mean().item()
                                step_log["val_loss"] = val_loss
                    finally:
                        if model_was_training:
                            self.model.train()
                        if self.ema_model is not None and ema_was_training:
                            self.ema_model.train()

                if (
                    self.epoch % cfg.training.sample_every
                ) == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]
                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    metric_dict["epoch"] = self.epoch
                    metric_dict["global_step"] = self.global_step
                    if cfg.checkpoint.save_all_ckpt:
                        ckpt_name = cfg.checkpoint.all_format_str.format(**metric_dict)
                        ckpt_path = Path(self.output_dir).joinpath(
                            "checkpoints", ckpt_name
                        )
                        self.save_checkpoint(path=ckpt_path)
                policy.train()
                self._log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()


@hydra.main(
    config_path=str(Path(__file__).parent.parent.joinpath("config")),
    config_name="train",
    version_base=None,
)
def main(cfg):
    workspace = RealWorldWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
