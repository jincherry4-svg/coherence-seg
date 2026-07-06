"""訓練進入點（規格書 §7、§13）。

用法：python -m src.train --config <里程碑config> [<設備config> ...] [--seed 42] [--resume]
例如：python -m src.train --config configs/m2_reorder.yaml configs/lab_1080ti.yaml
多個 config 由左至右依序疊加合併（後者覆寫前者）。
"""

from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from .data.collate import CurriculumController, SegmentationCollator, worker_init_fn
from .data.unlabeled import load_unlabeled
from .data.wikisection import SegDataset, build_special_ids, load_jsonl
from .models.lit_module import SegLitModule


def build_tokenizer(cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    tok.add_special_tokens({"additional_special_tokens": ["[SLOT]", "[CAND]"]})
    return tok


def build_loaders(cfg, tokenizer, ids, seed):
    train_ctrl = CurriculumController(enabled=cfg.flags.use_reorder)
    eval_ctrl = CurriculumController(enabled=False)  # 驗證/測試永不挖空
    common = dict(num_workers=cfg.data.num_workers, worker_init_fn=worker_init_fn,
                  pin_memory=True)

    def mk(dataset, shuffle, need_clean):
        return DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=shuffle,
                          collate_fn=SegmentationCollator(ids, need_clean=need_clean), **common)

    need_clean = bool(cfg.flags.use_mean_teacher)
    tr_docs = load_jsonl(cfg.data.train_path, max_docs=cfg.data.get("max_train_docs"))
    va_docs = load_jsonl(cfg.data.dev_path)
    te_docs = load_jsonl(cfg.data.test_path)
    mk_ds = lambda docs, ctrl: SegDataset(docs, tokenizer, ids, ctrl,
                                          max_len=cfg.data.max_len, base_seed=seed)
    loaders = {
        "train": mk(mk_ds(tr_docs, train_ctrl), True, need_clean),
        "val": mk(mk_ds(va_docs, eval_ctrl), False, False),
        "test": mk(mk_ds(te_docs, eval_ctrl), False, False),
    }
    if cfg.flags.use_unlabeled:
        un_docs = load_unlabeled(cfg.data.unlabeled_path, max_docs=cfg.data.get("max_unlabeled_docs"))
        # 無標註 loader 與 labeled 共用 curriculum
        loaders["unlabeled"] = mk(mk_ds(un_docs, train_ctrl), True, need_clean)
    return loaders, train_ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, nargs="+",
                    help="一個或多個 yaml，依序疊加（里程碑 config 在前、設備 config 在後）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    for c in args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(c))
    run_tag = "-".join(os.path.basename(c).replace(".yaml", "") for c in args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    pl.seed_everything(seed, workers=True)

    tokenizer = build_tokenizer(cfg)
    ids = build_special_ids(tokenizer)
    loaders, curriculum = build_loaders(cfg, tokenizer, ids, seed)

    module = SegLitModule(cfg, curriculum=curriculum, tokenizer_size=len(tokenizer))

    logger = True
    if cfg.wandb.enabled and os.environ.get("WANDB_API_KEY"):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=cfg.wandb.project, group=cfg.wandb.group,
                             name=f"{run_tag}-s{seed}",
                             config={"seed": seed, **OmegaConf.to_container(cfg)})

    ckpt_dir = os.path.join(cfg.ckpt_dir, run_tag, f"seed{seed}")
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, monitor="val/pk", mode="min", save_top_k=1,
                        save_last=True, every_n_train_steps=cfg.train.ckpt_every_n_steps),
        EarlyStopping(monitor="val/pk", mode="min", patience=cfg.train.patience),
    ]
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.grad_accum,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        deterministic="warn",
    )

    if cfg.flags.use_unlabeled:
        from pytorch_lightning.utilities import CombinedLoader

        train_loader = CombinedLoader(
            {"labeled": loaders["train"], "unlabeled": loaders["unlabeled"]},
            mode="max_size_cycle",
        )
    else:
        train_loader = loaders["train"]

    resume_path = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_path = resume_path if (args.resume and os.path.exists(resume_path)) else None
    trainer.fit(module, train_loader, loaders["val"], ckpt_path=ckpt_path)
    trainer.test(module, loaders["test"], ckpt_path="best")


if __name__ == "__main__":
    main()
