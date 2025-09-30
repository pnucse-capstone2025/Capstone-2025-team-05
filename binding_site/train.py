import yaml
import os
import logging
import sys
import pandas as pd
import numpy as np
import pickle
import argparse

from modules.utils import load_cfg
from modules.data import PocketDataset, Dataloader
from modules.TrainIters import PlanetXTrainIter

from sklearn.model_selection import KFold

def main():
    """ Define argparse """
    parser = argparse.ArgumentParser(
        description="PLaNet-X predicts binding site based on protein sequence."
    )
    parser.add_argument(
        "--config", "-c", required=True, type=input_check,
        help="The file contains information on the protein sequences to predict binding sites."
    )
    args = parser.parse_args()

    config = load_cfg(args.config)

    # 1) Load data
    print("1. Load data ...")
    with open(config["paths"]["prot_feats"], "rb") as f:
        IDs, sequences, binding_sites, protein_feats = pickle.load(f)

    IDs = np.array(IDs)
    sequences = np.array(sequences, dtype=object)
    binding_sites = np.array(binding_sites, dtype=object)
    protein_feats = np.array(protein_feats, dtype=object)

    # 2) K-Fold
    print("2. Run K-fold Cross Validation ...")
    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    for idx, (train_idx, val_index) in enumerate(kf.split(IDs)):
        train_IDs, train_feats, train_BS, train_seqs = (
            IDs[train_idx], protein_feats[train_idx], binding_sites[train_idx], sequences[train_idx]
        )
        validation_IDs, val_feats, val_BS, val_seqs = (
            IDs[val_index], protein_feats[val_index], binding_sites[val_index], sequences[val_index]
        )

        # dataset/dataloader
        train_dataset = PocketDataset(train_IDs, train_feats, train_seqs, train_BS)
        train_loader = Dataloader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)

        val_dataset = PocketDataset(validation_IDs, val_feats, val_seqs, val_BS)
        val_loader = Dataloader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

        save_dir = os.path.join(config["paths"]["save_path"], f"fold{idx}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"\tFold {idx} is running ...")
        trainiter = PlanetXTrainIter(config)
        trainiter.train(train_loader, val_loader, save_dir)

def input_check(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('%s does not exist.' % path)
    return path

if __name__ == "__main__":
    main()
