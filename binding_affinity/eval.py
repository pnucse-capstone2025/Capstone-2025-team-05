# eval.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from transformers import AutoModel

from planet_bap import ProteinLigandModel
from dataset import MyDataset
from metrics_utils import metrics_dict

# data_path = "/workspace/binding_affinity/datasets/validation2020"
# data_path = "/workspace/binding_affinity/datasets/validation2016"
# data_path = "/workspace/binding_affinity/datasets/train2020"
# data_path = "/workspace/binding_affinity/datasets/train2016"
# data_path = "/workspace/binding_affinity/datasets/CASF-2013"
# data_path = "/workspace/binding_affinity/datasets/CSAR-HiQ"
# data_path = "/workspace/binding_affinity/datasets/Core2016"
data_path = "/workspace/binding_affinity/datasets/train2020"
# model_path = "/workspace/binding_affinity/best2016/best_model.pt"
model_path = "/workspace/binding_affinity/best2020/best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    pt_seqs, smi_seqs, smi_masks, lig_globals, ys = zip(*batch)

    max_pt_len = max(x.size(0) for x in pt_seqs)
    pt_padded = torch.stack([F.pad(x, (0, 0, 0, max_pt_len - x.size(0))) for x in pt_seqs])

    max_smi_len = max(x.size(0) for x in smi_seqs)
    smi_padded = torch.stack([F.pad(x, (0, max_smi_len - x.size(0))) for x in smi_seqs])

    smi_masks_padded = torch.stack([F.pad(x, (0, max_smi_len - x.size(0))) for x in smi_masks])

    lig_globals = torch.stack(lig_globals)  
    ys = torch.stack(ys)                   

    return pt_padded, smi_padded, smi_masks_padded, lig_globals, ys

# test_dataset = MyDataset(data_path, split="train2016")
# test_dataset = MyDataset(data_path, split="validation2020")
# test_dataset = MyDataset(data_path, split="validation2016")
# test_dataset = MyDataset(data_path, split="CSAR36")
# test_dataset = MyDataset(data_path, split="CSAR51")
# test_dataset = MyDataset(data_path, split="Core2016")
# test_dataset = MyDataset(data_path, split="CASF-2013")
test_dataset = MyDataset(data_path, split="train2020")
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn
)

chembert_model_name = "seyonec/ChemBERTa-zinc-base-v1"
chembert_model = AutoModel.from_pretrained(chembert_model_name)

model = ProteinLigandModel(chembert_model=chembert_model).to(device)  
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for pt_seq, smi_seq, smi_mask, lig_global, y in test_loader:
        pt_seq, smi_seq, smi_mask, lig_global, y = (
            pt_seq.to(device), smi_seq.to(device),
            smi_mask.to(device), lig_global.to(device),
            y.to(device)
        )

        print("pt_seq:", pt_seq.shape)
        print("smi_seq:", smi_seq.shape)
        print("smi_mask:", smi_mask.shape)
        print("lig_global:", lig_global.shape)

        out = model(pt_seq, smi_seq, smi_mask, lig_global).view(-1)

        y_true.append(y.cpu().numpy())
        y_pred.append(out.cpu().numpy())

y_true = np.concatenate(y_true).astype(np.float32)
y_pred = np.concatenate(y_pred).astype(np.float32)

metrics = metrics_dict(y_true, y_pred)
print("=== Overall Test metrics ===")
for k, v in metrics.items():
    try:
        print(f"{k}: {v:.4f}")
    except Exception:
        print(f"{k}: {v}")

save_path = Path("./saveModel/test_results")
save_path.mkdir(parents=True, exist_ok=True)
np.savez(save_path / "test_metrics.npz", y_true=y_true, y_pred=y_pred, **metrics)

bin_edges = np.array([2.0, 4.0, 6.0, 8.0, 10.0, np.inf], dtype=np.float32)
bin_labels = ["[2, 4)", "[4, 6)", "[6, 8)", "[8, 10)", "[10, inf)"]

def safe_rmse(y_t, y_p):
    if y_t.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))

def safe_mae(y_t, y_p):
    if y_t.size == 0:
        return np.nan
    return float(np.mean(np.abs(y_t - y_p)))

def safe_max_err(y_t, y_p):
    if y_t.size == 0:
        return np.nan
    return float(np.max(np.abs(y_t - y_p)))

rows = []
print("\n--- Fixed-bin metrics ---")
print(f"{'Range':<12} {'Count':>6} {'RMSE':>8} {'MAE':>8} {'Max Error':>11}")
for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    if np.isfinite(hi):
        idx = np.where((y_true >= lo) & (y_true < hi))[0]
    else:
        idx = np.where(y_true >= lo)[0]

    y_t_bin = y_true[idx]
    y_p_bin = y_pred[idx]

    rmse = safe_rmse(y_t_bin, y_p_bin)
    mae  = safe_mae(y_t_bin, y_p_bin)
    mxe  = safe_max_err(y_t_bin, y_p_bin)
    cnt  = int(y_t_bin.size)

    rows.append({
        "range": bin_labels[i],
        "low": float(lo),
        "high": None if not np.isfinite(hi) else float(hi),
        "count": cnt,
        "rmse": rmse,
        "mae": mae,
        "max_error": mxe,
    })

    rmse_s = f"{rmse:.4f}" if rmse == rmse else "nan"
    mae_s  = f"{mae:.4f}"  if mae == mae   else "nan"
    mxe_s  = f"{mxe:.4f}"  if mxe == mxe   else "nan"
    print(f"{bin_labels[i]:<12} {cnt:>6} {rmse_s:>8} {mae_s:>8} {mxe_s:>11}")

import csv
csv_path = save_path / "bin_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["range", "low", "high", "count", "rmse", "mae", "max_error"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

np.savez(save_path / "bin_metrics.npz", **{
    "ranges": np.array([r["range"] for r in rows], dtype=object),
    "lows": np.array([r["low"] for r in rows], dtype=np.float32),
    "highs": np.array([np.inf if r["high"] is None else r["high"] for r in rows], dtype=np.float32),
    "counts": np.array([r["count"] for r in rows], dtype=np.int32),
    "rmses": np.array([np.nan if (r["rmse"] != r["rmse"]) else r["rmse"] for r in rows], dtype=np.float32),
    "maes": np.array([np.nan if (r["mae"]  != r["mae"])  else r["mae"]  for r in rows], dtype=np.float32),
    "max_errors": np.array([np.nan if (r["max_error"] != r["max_error"]) else r["max_error"] for r in rows], dtype=np.float32),
})
print(f"\nSaved overall metrics to: {save_path / 'test_metrics.npz'}")
print(f"Saved binned metrics CSV to: {csv_path}")
print(f"Saved binned metrics NPZ to: {save_path / 'bin_metrics.npz'}")
