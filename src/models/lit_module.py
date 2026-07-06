"""訓練核心 LightningModule（規格書 §6、§7）。

模式旗標（config）：
- use_reorder:      M2+ 開啟句子重組輔助任務
- use_unlabeled:    M3+ 混入無標註資料（CombinedLoader，key: "labeled"/"unlabeled"）
- use_mean_teacher: M4  開啟 EMA 教師與一致性損失
"""

from __future__ import annotations

from collections import deque

import pytorch_lightning as pl
import torch

from ..losses import (
    bce_with_pos_weight,
    binary_focal_loss_with_logits,
    consistency_mse,
    matching_accuracy,
    matching_cross_entropy,
    sigmoid_rampup,
)
from ..eval.metrics import pk_wd_spokennlp, scan_threshold
from .ema import EmaTeacher
from .encoder import SegmentationModel


class SegLitModule(pl.LightningModule):
    def __init__(self, cfg, curriculum=None, tokenizer_size: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.curriculum = curriculum
        self.model = SegmentationModel(
            model_name=cfg.model.name,
            tokenizer_size=tokenizer_size,
            dropout=cfg.model.dropout,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )
        self.teacher = (
            EmaTeacher(self.model, max_decay=cfg.mean_teacher.max_decay)
            if cfg.flags.use_mean_teacher
            else None
        )
        self._reorder_acc_window = deque(maxlen=cfg.curriculum.acc_window)
        self._val_probs: list[list[float]] = []
        self._val_refs: list[list[int]] = []
        self.save_hyperparameters({"cfg": dict(cfg)})

    # ---------- loss 組件 ----------

    def _seg_loss(self, logits, batch):
        if self.cfg.loss.seg_type == "focal":
            return binary_focal_loss_with_logits(
                logits, batch["seg_labels"], batch["seg_mask"],
                gamma=self.cfg.loss.focal_gamma, alpha=self.cfg.loss.focal_alpha,
            )
        return bce_with_pos_weight(
            logits, batch["seg_labels"], batch["seg_mask"], pos_weight=self.cfg.loss.pos_weight
        )

    def _forward_student(self, batch, compute_reorder: bool):
        return self.model(
            batch["input_ids"], batch["attention_mask"], batch["global_attention_mask"],
            batch["sent_anchor_idx"],
            batch.get("slot_idx") if compute_reorder else None,
            batch.get("cand_idx") if compute_reorder else None,
        )

    def _batch_losses(self, batch, labeled: bool):
        """回傳 loss dict。labeled=False 時不計 L_seg；
        且僅在 flags.unlabeled_reorder=True 時才對無標註 batch 計 L_reorder
        （M3 的無標註資料只透過一致性損失貢獻，M4 才全開）。"""
        compute_reorder = self.cfg.flags.use_reorder and (
            labeled or self.cfg.flags.get("unlabeled_reorder", False)
        )
        out = self._forward_student(batch, compute_reorder)
        losses = {}
        if labeled:
            losses["seg"] = self._seg_loss(out["boundary_logits"], batch)
        if compute_reorder and "match_scores" in out:
            losses["reorder"] = matching_cross_entropy(out["match_scores"], batch["match_labels"])
            acc, n = matching_accuracy(out["match_scores"], batch["match_labels"])
            if n > 0:
                self._reorder_acc_window.append(acc)
                self.log("train/reorder_acc", acc, prog_bar=True)
        if self.cfg.flags.use_mean_teacher:
            with torch.no_grad():
                t_out = self.teacher(
                    batch["clean_input_ids"], batch["clean_attention_mask"],
                    batch["clean_global_attention_mask"], batch["clean_sent_anchor_idx"],
                )
            valid = (batch["is_slot"] == 0) & (batch["sent_anchor_idx"] != -1)
            losses["consistency"] = consistency_mse(
                torch.sigmoid(out["boundary_logits"]),
                torch.sigmoid(t_out["boundary_logits"]),
                valid,
            )
        return losses

    # ---------- Lightning hooks ----------

    def training_step(self, batch, batch_idx):
        step = self.global_step
        lam1 = sigmoid_rampup(step, self.cfg.loss.lambda1_ramp_steps, self.cfg.loss.lambda1_max)
        lam2 = sigmoid_rampup(step, self.cfg.loss.lambda2_ramp_steps, self.cfg.loss.lambda2_max)

        if isinstance(batch, dict) and "labeled" in batch:  # CombinedLoader（M3+）
            labeled_batch, unlabeled_batch = batch["labeled"], batch.get("unlabeled")
        else:
            labeled_batch, unlabeled_batch = batch, None

        total = torch.tensor(0.0, device=self.device)
        if labeled_batch:
            l = self._batch_losses(labeled_batch, labeled=True)
            total = total + l.get("seg", 0.0) + lam1 * l.get("reorder", 0.0) \
                + lam2 * l.get("consistency", 0.0)
            for k, v in l.items():
                self.log(f"train/L_{k}", v)
        if unlabeled_batch:
            u = self._batch_losses(unlabeled_batch, labeled=False)
            total = total + lam1 * u.get("reorder", 0.0) + lam2 * u.get("consistency", 0.0)
            for k, v in u.items():
                self.log(f"train/L_{k}_unlabeled", v)

        self.log_dict({"train/lambda1": lam1, "train/lambda2": lam2,
                       "train/p_stage": float(self.curriculum.stage.value) if self.curriculum else 0.0})
        self._maybe_advance_curriculum()

        # 【防彈級安全鎖】防止特定 Batch 因無效標籤導致 total 失去 grad_fn 觸發暴斃
        if total.grad_fn is None:
            # 尋找模型中任何一個需要梯度的參數，將其乘上 0.0 注入 total
            # 這樣數值完全保持不變，但能強制幫 total 接回 PyTorch 的反向傳播計算圖中
            for param in self.parameters():
                if param.requires_grad:
                    total = total + (param.sum() * 0.0)
                    break
        
        return total

    def on_train_batch_end(self, *args, **kwargs):
        if self.teacher is not None:
            d = self.teacher.update(self.model, self.global_step)
            self.log("train/ema_decay", d)

    def _maybe_advance_curriculum(self):
        """重組準確率移動平均 > 門檻時升級挖空難度（§6），並清空視窗。"""
        if self.curriculum is None or not self.cfg.flags.use_reorder:
            return
        w = self._reorder_acc_window
        if len(w) == w.maxlen and sum(w) / len(w) > self.cfg.curriculum.acc_threshold:
            if self.curriculum.stage.value < 2:
                new_stage = self.curriculum.advance()
                w.clear()
                self.log("train/curriculum_stage", float(new_stage))

    def validation_step(self, batch, batch_idx):
        if not batch:
            return
        out = self.model(
            batch["input_ids"], batch["attention_mask"], batch["global_attention_mask"],
            batch["sent_anchor_idx"],
        )
        probs = torch.sigmoid(out["boundary_logits"])
        for b in range(probs.size(0)):
            valid = batch["sent_anchor_idx"][b] != -1
            labels = batch["seg_labels"][b][valid]
            keep = labels != -100
            self._val_probs.append(probs[b][valid][keep].float().cpu().tolist())
            self._val_refs.append(labels[keep].cpu().tolist())

    def on_validation_epoch_end(self):
        if not self._val_probs:
            return
        t, res = scan_threshold(self._val_probs, self._val_refs)
        self.log_dict({f"val/{k}": v for k, v in res.items()})
        self.log("val/best_threshold", t)
        self._val_probs, self._val_refs = [], []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.param_groups(self.cfg.optim.lr_encoder, self.cfg.optim.lr_heads,
                                    self.cfg.optim.weight_decay)
        )
        total = self.trainer.estimated_stepping_batches
        warmup = int(self.cfg.optim.warmup_ratio * total)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: s / max(warmup, 1) if s < warmup
            else max(0.0, (total - s) / max(total - warmup, 1)),
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # ---------- checkpoint 續訓（§9.3）----------

    def on_save_checkpoint(self, ckpt):
        if self.curriculum is not None:
            ckpt["curriculum"] = self.curriculum.state_dict()

    def on_load_checkpoint(self, ckpt):
        if self.curriculum is not None and "curriculum" in ckpt:
            self.curriculum.load_state_dict(ckpt["curriculum"])
