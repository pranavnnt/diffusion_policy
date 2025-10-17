import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy, random, wandb, tqdm, numpy as np, shutil

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    # ------------------------------------------------------------------ #
    # ✅ These two functions are what eval.py calls when re-creating
    #    the workspace from a checkpoint. They must exist.
    # ------------------------------------------------------------------ #
    def build_dataset(self):
        dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        return dataset

    def build_model(self):
        model = hydra.utils.instantiate(self.cfg.policy)
        return model
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # model + optimizer
        self.model: DiffusionTransformerHybridImagePolicy = self.build_model()
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        # datasets
        dataset = self.build_dataset()
        train_loader = DataLoader(dataset, **cfg.dataloader)
        val_loader = DataLoader(dataset.get_validation_dataset(), **cfg.val_dataloader)
        normalizer = dataset.get_normalizer()

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # scheduler + ema
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_loader)*cfg.training.num_epochs)//cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1,
        )
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if cfg.training.use_ema else None

        # env-runner (optional rollout)
        env_runner = None
        if hasattr(cfg.task, "env_runner") and cfg.task.env_runner is not None:
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
            # 🔧 corrected type check
            assert isinstance(env_runner, BaseImageRunner)

        # logging + checkpoint
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model: self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # debug overrides
        if cfg.training.debug:
            for k, v in dict(num_epochs=2, max_train_steps=3, max_val_steps=3,
                             rollout_every=1, checkpoint_every=1,
                             val_every=1, sample_every=1).items():
                cfg.training[k] = v

        # ---------------------- TRAIN LOOP ---------------------- #
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log, train_losses = {}, []
                with tqdm.tqdm(train_loader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        raw_loss = self.model.compute_loss(batch)
                        (raw_loss / cfg.training.gradient_accumulate_every).backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step(); self.optimizer.zero_grad(); lr_scheduler.step()
                        if ema: ema.step(self.model)

                        loss_val = raw_loss.item()
                        train_losses.append(loss_val)
                        step_log = {'train_loss': loss_val,
                                    'global_step': self.global_step,
                                    'epoch': self.epoch,
                                    'lr': lr_scheduler.get_last_lr()[0]}
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                        if cfg.training.max_train_steps and batch_idx >= cfg.training.max_train_steps-1:
                            break

                step_log['train_loss'] = float(np.mean(train_losses))
                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                # optional rollout
                if env_runner and (self.epoch % cfg.training.rollout_every == 0):
                    step_log.update(env_runner.run(policy))

                # validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = []
                        for batch_idx, batch in enumerate(val_loader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            val_losses.append(self.model.compute_loss(batch))
                            if cfg.training.max_val_steps and batch_idx >= cfg.training.max_val_steps-1:
                                break
                        if val_losses:
                            step_log['val_loss'] = float(torch.mean(torch.tensor(val_losses)))

                # checkpointing
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt: self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot: self.save_snapshot()
                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                    ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if ckpt_path: self.save_checkpoint(path=ckpt_path)

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        # ---------------------- END LOOP ---------------------- #


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
