import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from .planet_x import Planet_X
from .helpers import prepare_prots_input

class PlanetXTrainIter:
    def __init__(self, config):
        self.config = config

        # build model
        self.model = Planet_X(self.config).cuda()

        # optimizer
        self.optim = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=config["train"]["weight_decay"]
        )

        # ===== 기록 버킷 =====
        self.train_losses, self.val_losses = [], []
        self.train_precisions, self.val_precisions = [], []
        self.train_f1s, self.val_f1s = [], []

        # threshold (없으면 0.5)
        self.threshold = float(self.config.get("train", {}).get("threshold", 0.5))

    # ----- metric 계산 유틸 -----
    @staticmethod
    def _tp_fp_fn(logits, targets, mask, thresh: float):
        """
        logits, targets, mask: (B, L)
        thresh: 시그모이드 확률 임계값
        return: (tp, fp, fn) floats
        """
        mask = mask.float()
        probs = torch.sigmoid(logits)
        preds = (probs >= thresh).float() * mask
        t = (targets > 0.5).float() * mask

        tp = (preds * t).sum().item()
        fp = (preds * (1.0 - t)).sum().item()
        fn = ((1.0 - preds) * t).sum().item()
        return tp, fp, fn

    @staticmethod
    def _precision_recall_f1(tp, fp, fn, eps=1e-8):
        prec = tp / max(eps, tp + fp)
        rec  = tp / max(eps, tp + fn)
        f1   = (2 * prec * rec) / max(eps, (prec + rec))
        return prec, rec, f1

    # ----- 학습 -----
    def train(self, train_loader, validation_loader, save_path):
        self.model_save_path = save_path
        best_eval_loss = float("inf")

        for epoch in range(self.config["train"]["epochs"]):
            # ---- Train ----
            self.model.train()
            loss_sum = 0.0
            tp_sum, fp_sum, fn_sum = 0.0, 0.0, 0.0

            for batch in tqdm(train_loader):
                aa_feats, prot_feats, prot_masks, binding_sites, position_ids, chain_idx = \
                    prepare_prots_input(self.config, batch, training=True)

                self.optim.zero_grad()

                # forward
                _, pred_BS = self.model(
                    aa_feats, prot_feats, prot_masks, position_ids, chain_idx
                )

                # loss
                loss = self.masked_bce_with_logits(
                    logits=pred_BS, targets=binding_sites, mask=prot_masks
                )
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()

                # metric counts
                tp, fp, fn = self._tp_fp_fn(pred_BS.detach(), binding_sites, prot_masks, self.threshold)
                tp_sum += tp; fp_sum += fp; fn_sum += fn

            train_loss = loss_sum / max(1, len(train_loader))
            train_prec, train_rec, train_f1 = self._precision_recall_f1(tp_sum, fp_sum, fn_sum)

            # ---- Validation ----
            val_loss_sum, vtp, vfp, vfn = self.eval(validation_loader)
            val_loss = val_loss_sum / max(1, len(validation_loader))
            val_prec, val_rec, val_f1 = self._precision_recall_f1(vtp, vfp, vfn)

            # 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_precisions.append(train_prec)
            self.val_precisions.append(val_prec)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)

            print(
                f"Epoch {epoch}: "
                f"Train loss {train_loss:.4f} | Val loss {val_loss:.4f} || "
                f"P {train_prec:.4f}/{val_prec:.4f} | "
                f"F1 {train_f1:.4f}/{val_f1:.4f}"
            )

            # 체크포인트
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                print("  ↳ New best. Saving checkpoint...")
                self.save_checkpoint(
                    self.model_save_path, "Planet_X.pth",
                    state_dict=self.model.state_dict(),
                    optimizer=self.optim.state_dict()
                )

    # ----- 검증 -----
    def eval(self, loader):
        loss_sum = 0.0
        tp_sum, fp_sum, fn_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            self.model.eval()
            for batch in loader:
                aa_feats, prot_feats, prot_masks, binding_sites, position_ids, chain_idx = \
                    prepare_prots_input(self.config, batch, training=True)

                _, pred_BS = self.model(
                    aa_feats, prot_feats, prot_masks, position_ids, chain_idx
                )

                loss = self.masked_bce_with_logits(
                    logits=pred_BS, targets=binding_sites, mask=prot_masks
                )
                loss_sum += loss.item()

                tp, fp, fn = self._tp_fp_fn(pred_BS, binding_sites, prot_masks, self.threshold)
                tp_sum += tp; fp_sum += fp; fn_sum += fn

        return loss_sum, tp_sum, fp_sum, fn_sum

    # ----- 테스트  -----
    @torch.no_grad()
    def run_test(self, loader, ckpts):
        """
        ckpts: str 또는 List[str]
        각 경로의 'Planet_X.pth'를 순회 로드하여 예측 평균
        반환: np.ndarray (N, L) — residue별 확률(0~1)
        """
        if isinstance(ckpts, str):
            ckpts = [ckpts]
        norm_ckpts = []
        for p in ckpts:
            planet_ckpt = os.path.join(p, "Planet_X.pth") if os.path.isdir(p) else p
            if os.path.isdir(planet_ckpt):
                planet_ckpt = os.path.join(planet_ckpt, "Planet_X.pth")
            if not os.path.exists(planet_ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {planet_ckpt}")
            norm_ckpts.append(planet_ckpt)
        ckpts = norm_ckpts

        self.model.eval()
        all_probs = []

        for batch in loader:
            aa_feats, prot_feats, prot_masks, position_ids, chain_idx = \
                prepare_prots_input(self.config, batch, training=False)

            probs_sum = None
            for ckpt_path in ckpts:
                state = torch.load(
                    ckpt_path, map_location=next(self.model.parameters()).device
                )
                sd = state.get("state_dict", state.get("model", state))
                self.model.load_state_dict(sd, strict=False)

                _, logits = self.model(
                    aa_feats, prot_feats, prot_masks, position_ids, chain_idx
                )  # (B, L)
                probs = torch.sigmoid(logits) * prot_masks
                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            probs_mean = probs_sum / len(ckpts)
            all_probs.append(probs_mean.detach().cpu())

        return torch.cat(all_probs, dim=0).numpy()

    # ----- 손실 -----
    def masked_bce_with_logits(self, logits, targets, mask):
        """
        logits : (B, L)  - 모델 raw output (시그모이드 적용 X)
        targets: (B, L)  - 0/1 라벨
        mask   : (B, L)  - 유효 위치=1, 패딩=0
        """
        logits = logits.float()
        targets = targets.float()
        mask = mask.float()

        pos = (targets * mask).sum()
        neg = ((1.0 - targets) * mask).sum()
        pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)

        loss_el = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        loss_el = loss_el * mask
        denom = mask.sum().clamp(min=1.0)
        return loss_el.sum() / denom

    # ----- 저장 -----
    def save_checkpoint(self, dir, name, **kwargs):
        state = {}
        state.update(kwargs)
        filepath = os.path.join(dir, name)
        torch.save(state, filepath)
