"""訓練進入點（規格書 §7、§13）。

用法：python -m src.train --config <里程碑config> [<設備config> ...] [--seed 42] [--resume]
例如：python -m src.train --config configs/m2_reorder.yaml configs/lab_1080ti.yaml
多個 config 由左至右依序疊加合併（後者覆寫前者）。
"""

from __future__ import annotations

import argparse
import glob
import inspect
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


def _allowlist_omegaconf_globals():
    """PyTorch 2.6 起 torch.load 預設 weights_only=True，會拒絕反序列化 checkpoint 內的
    OmegaConf 物件。舊 checkpoint 的超參數夾帶 DictConfig，這裡明確允許這些「本專案自己
    寫入的可信類別」，作為 weights_only=False 之外的備援載入路徑。"""
    try:
        import torch.serialization as _ts
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode, ValueNode
        _ts.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Metadata,
                              AnyNode, ValueNode, dict, list])
    except Exception:
        pass


def _find_last_ckpt(ckpt_dir):
    """回傳 ckpt_dir 下最新的 last*.ckpt（含 Lightning 的 last-v1.ckpt 版本後綴），無則 None。"""
    cands = glob.glob(os.path.join(ckpt_dir, "last*.ckpt"))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def resolve_resume(cfg, exp_tag, seed, want_resume):
    """解析續訓 checkpoint，並把整個判斷過程印出來（§9.3）。

    回傳 (ckpt_path 或 None, ckpt_dir)。ckpt_dir 是本次訓練「應該」讀寫的資料夾，
    與 ModelCheckpoint 的 dirpath 一致，確保存檔與續訓永遠指向同一處。

    設計原則：
    - exp_tag = 里程碑 config + 資料集後綴（不含設備 config），所以同一實驗換設備、
      斷線重連，資料夾名都固定為 {ckpt_dir}/{里程碑[+資料集]}/seed{seed}，續訓才找得回來；
      但換資料集（例如 disease→city）視為不同實驗身分，各自獨立目錄，不會互撞。
    - 找不到時「大聲」印警告並掃描其他可能的根目錄，把真正的 checkpoint 位置攤在眼前，
      但**仍回傳 None 從頭開始**——因為 run_all_seeds.sh 對每個 seed 都無條件帶 --resume，
      報錯中止會害整批多 seed 掛掉。
    """
    ckpt_dir = os.path.abspath(os.path.join(cfg.ckpt_dir, exp_tag, f"seed{seed}"))
    resume_path = _find_last_ckpt(ckpt_dir)

    print("=" * 68)
    print(f"[resume] --resume = {want_resume}")
    print(f"[resume] 解析後的 ckpt_dir : {ckpt_dir}")
    print(f"[resume] 此目錄的 last*.ckpt: {resume_path or '（無）'}")

    if not want_resume:
        print("[resume] 未指定 --resume → 從頭訓練（step 0）")
        print("=" * 68)
        return None, ckpt_dir

    if resume_path:
        size_gb = os.path.getsize(resume_path) / 1e9
        print(f"[resume] ✅ 將從此 checkpoint 續訓（{size_gb:.2f} GB）：{resume_path}")
        print("=" * 68)
        return resume_path, ckpt_dir

    # 指定了 --resume 卻沒找到 → 掃描其他常見根目錄，把真正的位置指出來
    print("[resume] ⚠️  指定了 --resume，但上述目錄找不到 last.ckpt！")
    print("[resume] ⚠️  將從頭開始（step 0）。若你預期要續訓，請看以下掃描結果對齊路徑：")
    seen = set()
    hits = []
    candidate_roots = [
        cfg.ckpt_dir,
        os.environ.get("CKPT_DIR", ""),
        "./checkpoints",
        "/content/drive/MyDrive/coherence-seg/checkpoints",
        "/content/drive/MyDrive/LongformerSC/coherence-seg/checkpoints",
    ]
    for root in candidate_roots:
        if not root:
            continue
        root_abs = os.path.abspath(root)
        if root_abs in seen or not os.path.isdir(root_abs):
            continue
        seen.add(root_abs)
        for f in glob.glob(os.path.join(root_abs, "**", f"seed{seed}", "last*.ckpt"),
                           recursive=True):
            hits.append(f)
    if hits:
        print("[resume] 🔎 在其他位置找到符合本 seed 的 checkpoint：")
        for f in sorted(set(hits)):
            print(f"           - {f}  ({os.path.getsize(f)/1e9:.2f} GB)")
        print("[resume] 👉 若要用它續訓，讓 ckpt_dir 對齊該根目錄，例如：")
        print("           export CKPT_DIR=<那個根>  或  改用對應的設備 config（1080 Ti→./checkpoints，A100→Drive）")
        print("           注意 §9.5：續訓務必用與存檔時相同的設備/精度，勿跨 fp16↔bf16 續訓。")
    else:
        print(f"[resume] 🔎 掃描過的根目錄都沒有 seed{seed} 的 last.ckpt。確認 Drive 已掛載、且此 seed 之前真的有跑過。")
    print("=" * 68)
    return None, ckpt_dir


def compute_exp_tag(config_paths: list[str], data_name: str = "disease") -> str:
    """由 --config 疊加序列 + 合併後 cfg.data.name 算出實驗身分標籤。

    規則：milestone（config_paths[0]，依慣例里程碑一律排第一）+ 資料集後綴。
    資料集後綴的**唯一真相來源是合併後的 cfg.data.name**：
    - data_name == "disease"（預設）→ 不加後綴，與既有 disease 結果/checkpoint 相容
    - 其他（如 "city"）→ 附加 "+city"
    設備 config（如 lab_1080ti.yaml）不影響 exp_tag。

    範例：
        (["configs/m2_reorder.yaml"], "disease") → "m2_reorder"
        (["configs/m2_reorder.yaml", "configs/data_city.yaml"], "city") → "m2_reorder+city"
    """
    milestone_tag = os.path.basename(config_paths[0]).replace(".yaml", "")
    if data_name and data_name != "disease":
        return f"{milestone_tag}+{data_name}"
    return milestone_tag


def print_data_diagnostics(cfg):
    """開機大聲印出「本次到底用哪個資料集」，並交叉檢查 name 與路徑是否一致。

    背景：曾發生操作者以為在跑 city、實際讀的是 disease 路徑，兩份結果數字
    幾乎相同才驚覺（Pk 17.32 vs 17.39）。
    """
    name = cfg.data.get("name", "（未設定！請在 config 的 data.name 指定）")
    print("=" * 68)
    print(f"[data] 資料集身分（cfg.data.name）：{name}")
    for k in ("train_path", "dev_path", "test_path"):
        p = cfg.data.get(k)
        exists = "✅ 存在" if (p and os.path.isfile(p)) else "❌ 找不到檔案"
        print(f"[data] {k:10s}: {p}  {exists}")
    tp = cfg.data.get("train_path")
    if tp and os.path.isfile(tp):
        try:
            import json as _json
            with open(tp) as f:
                first = _json.loads(f.readline())
            uniq = sorted(set(int(v) for v in first.get("labels", [])))
            print(f"[data] train 首篇：{len(first.get('sentences', []))} 句，標籤獨特值 {uniq}"
                  f"（合法：-100/0/1 的子集）")
        except Exception as e:
            print(f"[data] ⚠️ 首篇標籤檢查失敗：{e}")
    if isinstance(name, str) and tp and name not in tp:
        print(f"[data] ⚠️⚠️ 警告：data.name = '{name}' 沒出現在 train_path 字串中！")
        print(f"[data] ⚠️⚠️ 極可能是「以為在跑 {name}、實際讀別的資料集」——請停下確認 config。")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, nargs="+",
                    help="一個或多個 yaml，依序疊加（里程碑 config 在前、設備 config 在後）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="不訓練：載入既有 checkpoint，重跑 validate（重掃門檻）與 test。"
                         "用途：改了評估邏輯後，不必重訓即可重測。")
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    for c in args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(c))
    # CKPT_DIR 環境變數優先於 config（與 run_all_seeds.sh 的 MARK_DIR 一致）。
    # 用途：checkpoint 存到了非預設根目錄時，一行 export 即可讓存檔與續訓對齊，免改 yaml。
    if os.environ.get("CKPT_DIR"):
        cfg.ckpt_dir = os.environ["CKPT_DIR"]
    # run_tag：完整（含設備）→ 只用於 wandb 顯示，看得出這個 run 跑在哪個設備。
    # exp_tag：里程碑 + 資料集後綴（見 compute_exp_tag，依 cfg.data.name 判斷）
    #          → 用於 checkpoint 資料夾與 results JSON 命名。設備不同不該改變
    #          實驗身分；資料集不同則視為不同實驗身分，各自獨立、不互撞。
    run_tag = "-".join(os.path.basename(c).replace(".yaml", "") for c in args.config)
    exp_tag = compute_exp_tag(args.config, cfg.data.get("name", "disease"))
    seed = args.seed if args.seed is not None else cfg.seed
    # 開機診斷：大聲印出本次使用的資料集與路徑
    print_data_diagnostics(cfg)
    # 【重要】把實驗身分注入 cfg，供 lit_module 的 on_test_epoch_end 寫 results JSON 使用：
    # - cfg.config_name：不注入的話會退回 cfg.model.name（例如 longformer-base-4096），
    #   M0/M2/M3/M4 的結果檔名撞在一起互相覆蓋，aggregate 無法分辨里程碑。
    # - cfg.seed：不回寫的話，--seed 43/44/... 跑出來的 JSON 仍讀 base.yaml 的預設 seed，
    #   五個 seed 的結果全寫到同一個 _seed42.json 互相覆蓋。
    cfg.config_name = exp_tag
    cfg.seed = seed
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

    # 先解析續訓：回傳的 ckpt_dir 同時作為 ModelCheckpoint 的 dirpath，
    # 保證「存檔目錄」與「續訓讀取目錄」永遠是同一個。
    ckpt_path, ckpt_dir = resolve_resume(cfg, exp_tag, seed, args.resume)
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

    # 續訓載入：這顆 checkpoint 是使用者自己訓練的可信檔案，故用 weights_only=False
    # 讓 torch.load 能還原內含的 OmegaConf 超參數（PyTorch 2.6 預設 True 會擋）。
    # 先註冊 OmegaConf 安全類別作為備援；再依 Lightning 版本決定是否傳 weights_only。
    _allowlist_omegaconf_globals()

    if args.eval_only:
        # 評估模式：不訓練。找 ckpt_dir 裡的 best（非 last 的最新 .ckpt）；
        # 找不到才退回 last*.ckpt。流程：validate（用當前程式碼的 scan_threshold
        # 重掃門檻）→ test（沿用 validate 已載入的權重，不重載，避免舊門檻蓋回）。
        all_ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        best_cands = [f for f in all_ckpts if not os.path.basename(f).startswith("last")]
        eval_ckpt = (max(best_cands, key=os.path.getmtime) if best_cands
                     else _find_last_ckpt(ckpt_dir))
        assert eval_ckpt, f"--eval-only 但 {ckpt_dir} 沒有任何 .ckpt 可載入"
        print(f"[eval-only] 載入 checkpoint：{eval_ckpt}"
              f"（{os.path.getsize(eval_ckpt)/1e9:.2f} GB，"
              f"{'best' if best_cands else 'last（找不到 best，退而求其次）'}）")
        v_kwargs = {}
        if "weights_only" in inspect.signature(trainer.validate).parameters:
            v_kwargs["weights_only"] = False
        trainer.validate(module, loaders["val"], ckpt_path=eval_ckpt, **v_kwargs)
        trainer.test(module, loaders["test"])
        return

    fit_kwargs = {}
    if "weights_only" in inspect.signature(trainer.fit).parameters:
        fit_kwargs["weights_only"] = False
    trainer.fit(module, train_loader, loaders["val"], ckpt_path=ckpt_path, **fit_kwargs)
    test_kwargs = {}
    if "weights_only" in inspect.signature(trainer.test).parameters:
        test_kwargs["weights_only"] = False
    trainer.test(module, loaders["test"], ckpt_path="best", **test_kwargs)


if __name__ == "__main__":
    main()
