"""Plain PyTorch training loop optimized for Kaggle sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import gc
import time
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from spine_xnet.constants import COLS, SEVERITY_CLASS_WEIGHTS
from spine_xnet.evaluation.metrics import classification_metrics, per_group_metrics, softmax_np
from spine_xnet.models.losses import LossWeights, SpineLoss
from spine_xnet.training.scheduler import warmup_cosine_scheduler
from spine_xnet.utils import AverageMeter, ensure_dir, to_device, unwrap_model, write_json


def compute_class_weights(labels: np.ndarray) -> list[float]:
    counts = np.bincount(labels.astype(int), minlength=3).astype(np.float32)
    inv = counts.sum() / np.clip(counts, 1.0, None)
    inv = inv / inv.mean()
    return inv.tolist()


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(cfg.get("lr", 1e-4))
    backbone_lr = float(cfg.get("backbone_lr", lr))
    concept_head_lr = float(cfg.get("concept_head_lr", lr * 3.0))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    backbone_params = []
    head_params = []
    concept_proto_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in ["backbone", "features"]):
            backbone_params.append(param)
        elif any(token in name for token in ["concept_head", "prototype_layer", "prototypes"]):
            concept_proto_params.append(param)
        else:
            head_params.append(param)
    groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": lr},
    ]
    if concept_proto_params:
        groups.append({"params": concept_proto_params, "lr": concept_head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        config: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        self.config = config
        self.output_dir = ensure_dir(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels_last = bool(config.get("training", {}).get("channels_last", True)) and self.device.type == "cuda"

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if torch.cuda.device_count() > 1 and config.get("training", {}).get("data_parallel", True):
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if bool(config.get("training", {}).get("compile", False)) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        train_cfg = config.get("training", {})
        batch_size = int(train_cfg.get("batch_size", 16))
        val_batch_size = int(train_cfg.get("val_batch_size", batch_size))
        num_workers = int(train_cfg.get("num_workers", 4))
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", False))
            loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        self.optimizer = build_optimizer(self.model, train_cfg)
        self.scheduler = warmup_cosine_scheduler(
            self.optimizer,
            warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
            max_epochs=int(train_cfg.get("epochs", 50)),
            steps_per_epoch=max(1, len(self.train_loader)),
        )

        loss_cfg = config.get("loss", {})
        weights = LossWeights(
            classification=float(loss_cfg.get("classification", 1.0)),
            concept=float(loss_cfg.get("concept", 0.3)),
            prototype_cluster=float(loss_cfg.get("prototype_cluster", 0.05)),
            prototype_diversity=float(loss_cfg.get("prototype_diversity", 0.02)),
            ordinal=float(loss_cfg.get("ordinal", 0.2)),
        )
        labels = train_dataset.df[COLS.label].values
        class_weights = loss_cfg.get("class_weights", None)
        if class_weights == "auto":
            class_weights = compute_class_weights(labels)
        elif class_weights is None:
            class_weights = SEVERITY_CLASS_WEIGHTS
        self.criterion = SpineLoss(
            class_weights=class_weights,
            loss_weights=weights,
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        ).to(self.device)

        self.grad_accum = int(train_cfg.get("grad_accum", 1))
        self.epochs = int(train_cfg.get("epochs", 50))
        self.patience = int(train_cfg.get("early_stopping_patience", 10))
        self.amp_enabled = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.log_every = int(train_cfg.get("log_every", 25))
        self.save_val_predictions = bool(train_cfg.get("save_val_predictions", False))
        self.save_val_predictions_best_only = bool(train_cfg.get("save_val_predictions_best_only", True))
        self.compute_group_metrics = bool(train_cfg.get("compute_group_metrics", True))
        self.progress_bar = bool(train_cfg.get("progress_bar", False))
        self.progress_mininterval = float(train_cfg.get("progress_mininterval", 30.0))
        self.validate_every = int(train_cfg.get("validate_every", 1))
        self.skip_validation = bool(train_cfg.get("skip_validation", False))
        self.max_train_batches = _optional_int(train_cfg.get("max_train_batches"))
        self.max_val_batches = _optional_int(train_cfg.get("max_val_batches"))
        self.checkpoint_every_steps = int(train_cfg.get("checkpoint_every_steps", 500))
        self.checkpoint_every_minutes = float(train_cfg.get("checkpoint_every_minutes", 20.0))
        self.heartbeat_every_steps = int(train_cfg.get("heartbeat_every_steps", 100))
        self.heartbeat_every_minutes = float(train_cfg.get("heartbeat_every_minutes", 5.0))
        self.time_limit_minutes = _optional_float(train_cfg.get("time_limit_minutes"))
        self.fit_start_time = time.monotonic()
        self.stop_requested = False
        self.best_metric = float("inf")
        self.start_epoch = 0

    def fit(self, resume: str | Path | None = None) -> dict[str, Any]:
        if resume:
            self.load_checkpoint(resume)

        history = []
        stale_epochs = 0
        print(
            {
                "event": "training_start",
                "epochs": self.epochs,
                "train_batches_per_epoch": len(self.train_loader),
                "val_batches": len(self.val_loader),
                "batch_size": self.train_loader.batch_size,
                "val_batch_size": self.val_loader.batch_size,
                "grad_accum": self.grad_accum,
                "amp": self.amp_enabled,
                "channels_last": self.channels_last,
                "device": str(self.device),
                "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "label_smoothing": float(self.config.get("loss", {}).get("label_smoothing", 0.0)),
                "residual_warmup_epochs": getattr(unwrap_model(self.model), "residual_warmup_epochs", 0),
            },
            flush=True,
        )
        for epoch in range(self.start_epoch, self.epochs):
            # Update residual weight annealing
            raw_model = unwrap_model(self.model)
            if hasattr(raw_model, "set_residual_weight_for_epoch"):
                raw_model.set_residual_weight_for_epoch(epoch)
            train_metrics = self.train_one_epoch(epoch)
            should_validate = (
                not self.skip_validation
                and self.validate_every > 0
                and ((epoch + 1) % self.validate_every == 0 or epoch == self.epochs - 1 or self.stop_requested)
            )
            if should_validate:
                val_metrics, pred_df = self.validate(epoch, return_predictions=self.save_val_predictions)
            else:
                val_metrics, pred_df = {}, None
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
            print(row, flush=True)

            monitor = float(val_metrics.get("weighted_log_loss", val_metrics.get("loss", train_metrics.get("total", float("inf")))))
            is_best = monitor < self.best_metric
            if is_best:
                self.best_metric = monitor
                stale_epochs = 0
            else:
                stale_epochs += 1

            self.save_checkpoint(epoch, is_best=is_best, metrics=val_metrics)
            write_json(row, self.output_dir / "latest_metrics.json")
            if self.save_val_predictions and pred_df is not None:
                if not self.save_val_predictions_best_only or is_best:
                    pred_df.to_csv(self.output_dir / f"val_predictions_epoch_{epoch:03d}.csv", index=False)
                del pred_df
            gc.collect()

            if self.stop_requested:
                print({"event": "time_limit_stop", "epoch": epoch, "elapsed_minutes": self.elapsed_minutes()}, flush=True)
                break
            if should_validate and stale_epochs >= self.patience:
                break
        return {"best_weighted_log_loss": self.best_metric, "epochs_ran": len(history)}

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        meters = {name: AverageMeter() for name in ["total", "classification", "concept", "prototype_cluster", "prototype_diversity", "ordinal"]}
        pbar = tqdm(
            self.train_loader,
            desc=f"train {epoch}",
            leave=False,
            disable=not self.progress_bar,
            mininterval=self.progress_mininterval,
        )
        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.monotonic()
        last_checkpoint_time = epoch_start
        last_heartbeat_time = epoch_start
        last_loss = float("nan")

        for step, batch in enumerate(pbar):
            if self.max_train_batches is not None and step >= self.max_train_batches:
                print({"event": "max_train_batches_reached", "epoch": epoch, "step": step}, flush=True)
                break
            batch = to_device(batch, self.device)
            if self.channels_last:
                batch["image"] = batch["image"].to(memory_format=torch.channels_last)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch["image"], batch["condition_idx"], batch["level_idx"])
                losses = self.criterion(outputs, batch, model=unwrap_model(self.model))
                loss = losses["total"] / self.grad_accum

            self.scaler.scale(loss).backward()
            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("training", {}).get("max_grad_norm", 1.0)))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            if step % self.log_every == 0:
                bs = batch["image"].shape[0]
                for name, value in losses.items():
                    meters[name].update(float(value.detach().cpu()), bs)
                last_loss = meters["total"].avg
                pbar.set_postfix(loss=meters["total"].avg)

            now = time.monotonic()
            if self._should_emit_heartbeat(step, now, last_heartbeat_time):
                last_heartbeat_time = now
                print(
                    {
                        "event": "heartbeat",
                        "epoch": epoch,
                        "step": step + 1,
                        "total_steps": len(self.train_loader),
                        "elapsed_minutes": round(self.elapsed_minutes(), 2),
                        "epoch_minutes": round((now - epoch_start) / 60.0, 2),
                        "loss": round(float(last_loss), 5) if np.isfinite(last_loss) else None,
                        "residual_weight": round(getattr(unwrap_model(self.model), '_residual_weight', 1.0), 3),
                    },
                    flush=True,
                )

            if self._should_step_checkpoint(step, now, last_checkpoint_time):
                last_checkpoint_time = now
                self.save_checkpoint(
                    epoch,
                    is_best=False,
                    metrics={"train_total": float(last_loss) if np.isfinite(last_loss) else None, "step": step + 1},
                    filename="latest_step.pt",
                    mid_epoch=True,
                )
                write_json(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "elapsed_minutes": self.elapsed_minutes(),
                        "train_total": float(last_loss) if np.isfinite(last_loss) else None,
                    },
                    self.output_dir / "latest_step.json",
                )

            if self._time_exceeded():
                self.stop_requested = True
                self.save_checkpoint(
                    epoch,
                    is_best=False,
                    metrics={"train_total": float(last_loss) if np.isfinite(last_loss) else None, "step": step + 1},
                    filename="time_limit.pt",
                    mid_epoch=True,
                )
                break
            del outputs, losses, loss, batch

        return {name: meter.avg for name, meter in meters.items()}

    @torch.no_grad()
    def validate(self, epoch: int = 0, return_predictions: bool = False) -> tuple[dict[str, float], pd.DataFrame | None]:
        self.model.eval()
        losses = AverageMeter()
        logits_list = []
        labels_list = []
        condition_list = []
        level_list = []
        rows = [] if return_predictions else None

        for step, batch in enumerate(tqdm(
            self.val_loader,
            desc=f"val {epoch}",
            leave=False,
            disable=not self.progress_bar,
            mininterval=self.progress_mininterval,
        )):
            if self.max_val_batches is not None and step >= self.max_val_batches:
                print({"event": "max_val_batches_reached", "epoch": epoch, "step": step}, flush=True)
                break
            batch = to_device(batch, self.device)
            if self.channels_last:
                batch["image"] = batch["image"].to(memory_format=torch.channels_last)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch["image"], batch["condition_idx"], batch["level_idx"])
                loss_dict = self.criterion(outputs, batch, model=unwrap_model(self.model))
            losses.update(float(loss_dict["total"].detach().cpu()), batch["image"].shape[0])
            logits = outputs["logits"].detach().float().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            logits_list.append(logits)
            labels_list.append(labels)
            if self.compute_group_metrics:
                condition_list.extend([str(c) for c in batch["condition"]])
                level_list.extend([str(l) for l in batch["level"]])
            probs = softmax_np(logits)
            if rows is not None:
                for i, sample_id in enumerate(batch["sample_id"]):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "study_id": int(batch["study_id"][i].detach().cpu()),
                            "condition": batch["condition"][i],
                            "level": batch["level"][i],
                            "label": int(labels[i]),
                            "prob_normal_mild": float(probs[i, 0]),
                            "prob_moderate": float(probs[i, 1]),
                            "prob_severe": float(probs[i, 2]),
                            "pred": int(probs[i].argmax()),
                        }
                    )
            del outputs, loss_dict, batch

        logits_arr = np.concatenate(logits_list)
        labels_arr = np.concatenate(labels_list)
        probs = softmax_np(logits_arr)
        metrics = classification_metrics(labels_arr, probs)
        metrics["loss"] = losses.avg
        if self.compute_group_metrics and condition_list:
            metrics.update(per_group_metrics(labels_arr, probs, np.asarray(condition_list), "condition"))
            metrics.update(per_group_metrics(labels_arr, probs, np.asarray(level_list), "level"))
        pred_df = pd.DataFrame(rows) if rows is not None else None
        return metrics, pred_df

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool,
        metrics: dict[str, float | int | None],
        filename: str = "last.pt",
        mid_epoch: bool = False,
    ) -> None:
        payload = {
            "epoch": epoch,
            "mid_epoch": mid_epoch,
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(payload, self.output_dir / filename)
        if is_best:
            torch.save(payload, self.output_dir / "best.pt")

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_metric = float(ckpt.get("best_metric", self.best_metric))
        self.start_epoch = int(ckpt.get("epoch", -1)) + (0 if ckpt.get("mid_epoch", False) else 1)

    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.fit_start_time) / 60.0

    def _time_exceeded(self) -> bool:
        return self.time_limit_minutes is not None and self.elapsed_minutes() >= self.time_limit_minutes

    def _should_emit_heartbeat(self, step: int, now: float, last_time: float) -> bool:
        if self.heartbeat_every_steps > 0 and (step + 1) % self.heartbeat_every_steps == 0:
            return True
        return self.heartbeat_every_minutes > 0 and (now - last_time) >= self.heartbeat_every_minutes * 60.0

    def _should_step_checkpoint(self, step: int, now: float, last_time: float) -> bool:
        if self.checkpoint_every_steps > 0 and (step + 1) % self.checkpoint_every_steps == 0:
            return True
        return self.checkpoint_every_minutes > 0 and (now - last_time) >= self.checkpoint_every_minutes * 60.0


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "none", "None"):
        return None
    return float(value)
