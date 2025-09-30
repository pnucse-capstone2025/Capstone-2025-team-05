import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_path, split="train"):
        """
        data_path : pkl 파일이 있는 폴더
        split : "train" 또는 "val"
        """
        file_path = f"{data_path}/{split}.pkl"
        with open(file_path, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pt_seq = torch.from_numpy(sample["prot_feats"]).float()

        smi_seq = torch.from_numpy(sample["lig_token_ids"]).long()
        smi_mask = torch.from_numpy(sample["lig_mask"]).bool()

        lig_global = torch.from_numpy(sample["lig_global"]).float()

        y = torch.tensor(sample["y"]).float()

        return pt_seq, smi_seq, smi_mask, lig_global, y
