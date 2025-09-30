import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_descriptor")
except RuntimeError:
    pass

class PLDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.from_numpy(sample["prot_feats"]).float(),  
            torch.from_numpy(sample["lig_token_ids"]).long(),
            torch.from_numpy(sample["lig_mask"]).bool(),     
            torch.from_numpy(sample["lig_global"]).float(),  
            torch.tensor(sample["y"]).float()                
        )


from planet_bap import ProteinLigandModel, LigandEmbedder  

train_dir = "../binding_affinity/datasets/train2020/"
val_dir = "../binding_affinity/datasets/core2016/"
train_pkl = f"{train_dir}train2020.pkl"
val_pkl   = f"{val_dir}core2016.pkl"

save_path = Path(f"./saveModel/planet_bap_{datetime.now().strftime('%Y%m%d%H%M%S')}")
save_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
n_epoch = 100
lr = 1e-4


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

train_dataset = PLDataset(train_pkl)
val_dataset   = PLDataset(val_pkl)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,             
    pin_memory=False,         
    persistent_workers=False,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    collate_fn=collate_fn,
)

chembert_model_name = "seyonec/ChemBERTa-zinc-base-v1"
chembert_model = AutoModel.from_pretrained(chembert_model_name)

model = ProteinLigandModel(chembert_model=chembert_model).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.SmoothL1Loss(beta=1.0) 

writer = SummaryWriter(save_path / "tensorboard")
best_val_loss = float("inf")
best_epoch = -1

top_k = 10
best_models = [] 

for epoch in range(1, n_epoch + 1):
    model.train()
    total_loss = 0.0
    for pt_seq, smi_seq, smi_mask, lig_global, y in tqdm(train_loader):
        pt_seq, smi_seq, smi_mask, lig_global, y = (
            pt_seq.to(device), smi_seq.to(device),
            smi_mask.to(device), lig_global.to(device),
            y.to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        out = model(pt_seq, smi_seq, smi_mask, lig_global).view(-1)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pt_seq.size(0)
    
    avg_train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for pt_seq, smi_seq, smi_mask, lig_global, y in val_loader:
            pt_seq, smi_seq, smi_mask, lig_global, y = (
                pt_seq.to(device), smi_seq.to(device),
                smi_mask.to(device), lig_global.to(device),
                y.to(device)
            )
            out = model(pt_seq, smi_seq, smi_mask, lig_global).view(-1)
            val_loss += loss_fn(out, y).item() * pt_seq.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))

    writer.add_scalar("train_loss", avg_train_loss, epoch)
    writer.add_scalar("val_loss", val_loss, epoch)
    writer.add_scalar("val_rmse", rmse, epoch)
    writer.add_scalar("val_mae", mae, epoch)

    with open(save_path / "metrics.csv", "a") as f:
        f.write(f"{epoch},{avg_train_loss},{val_loss},{rmse},{mae}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), save_path / "best_model.pt")

    model_path = save_path / f"epoch{epoch:03d}_loss{val_loss:.4f}.pt"
    torch.save(model.state_dict(), model_path)
    best_models.append((val_loss, epoch, model_path))

    best_models = sorted(best_models, key=lambda x: x[0])[:top_k]

    existing_paths = set(m[2] for m in best_models)
    for _, _, path in list(best_models):
        existing_paths.add(path)
    for f in save_path.glob("epoch*.pt"):
        if f not in existing_paths:
            f.unlink()

    print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
          f"rmse={rmse:.4f}, mae={mae:.4f}, best_epoch={best_epoch}")

print("Training finished. Best epoch:", best_epoch)
print("Top-10 saved models:", [f"epoch{e} (loss={l:.4f})" for l,e,_ in best_models])