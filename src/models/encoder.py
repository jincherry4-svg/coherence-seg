"""Longformer 編碼器包裝與完整學生模型（規格書 §5.1–§5.3）。

架構參考 SpokenNLP emnlp2023-topic_segmentation 的 longformer_for_ts.py：
以每句句首的 <s>（bos）token 作為句子表徵錨點；差異在於本專案的輔助任務
是槽位–候選配對（取代其 TSSP 的相鄰句序二元分類），且分類頭為單 logit
sigmoid（其為 2 類 CE，class 0 = 邊界）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .heads import BoundaryHead, MatchingHead, gather_positions


class SegmentationModel(nn.Module):
    """共享 Longformer 編碼器 + 邊界分類頭 + 句子重組配對頭。"""

    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        tokenizer_size: int | None = None,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        from transformers import LongformerModel  # 延遲匯入，離線測試不需要

        self.encoder = LongformerModel.from_pretrained(model_name, add_pooling_layer=False)
        if tokenizer_size is not None:  # §12 陷阱 1：加入 [SLOT]/[CAND] 後必須 resize
            self.encoder.resize_token_embeddings(tokenizer_size)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        d = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.boundary_head = BoundaryHead(d, dropout)
        self.matching_head = MatchingHead(d)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
        sent_anchor_idx: torch.Tensor,
        slot_idx: torch.Tensor | None = None,
        cand_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        ).last_hidden_state
        hidden = self.dropout(hidden)
        sent_repr = gather_positions(hidden, sent_anchor_idx)  # (B, S, d)
        out = {"boundary_logits": self.boundary_head(sent_repr)}
        if slot_idx is not None and cand_idx is not None and slot_idx.size(1) > 0:
            out["match_scores"] = self.matching_head(hidden, slot_idx, cand_idx)
        return out

    def param_groups(self, lr_encoder: float, lr_heads: float, weight_decay: float = 0.01):
        """optimizer 的兩組參數（§7）：encoder 低 LR、新頭高 LR。"""
        return [
            {"params": self.encoder.parameters(), "lr": lr_encoder, "weight_decay": weight_decay},
            {
                "params": list(self.boundary_head.parameters())
                + list(self.matching_head.parameters()),
                "lr": lr_heads,
                "weight_decay": weight_decay,
            },
        ]
