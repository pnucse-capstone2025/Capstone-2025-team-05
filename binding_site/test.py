# test.py
import yaml
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

from modules.utils import load_cfg
from modules.data import PocketDataset, Dataloader
from modules.TrainIters import PlanetXTrainIter
from modules.helpers import *  

def convert_bs(pred_binding_sites, threshold=0.6):
    """
    pred_binding_sites: (B, L) 확률 배열 (np.ndarray or torch.Tensor)
    return: ["1,5,9", "2,7", ...] 형태의 문자열 리스트
    """

    if pred_binding_sites is None:
        return []

    if isinstance(pred_binding_sites, torch.Tensor):
        pred = pred_binding_sites.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_binding_sites)

    if pred.ndim != 2:
        raise ValueError(f"convert_bs expects 2D array (B, L), got shape {pred.shape}")

    B, L = pred.shape
    results = []
    for b in range(B):
        scores = pred[b]
        idx = np.where(scores >= threshold)[0].tolist()
        results.append(",".join(map(str, idx)))
    return results

def get_results(binding_sites, pred_binding_sites, sequences, eps=1e-8):
    """
    binding_sites: 정답 문자열 리스트 (예: "3,5,6|10,11")
    pred_binding_sites: 예측 문자열 리스트 (예: "3,6,10")
    sequences: 각 단백질 서열(체인은 콤마로 이어진 문자열)
    """
    T_TP = T_TN = T_FP = T_FN = 0

    for bs, bps, seq in zip(binding_sites, pred_binding_sites, sequences):
        # 쉼표(,)는 체인 구분이라 길이 계산에서 제외
        seq_len = len([ch for ch in seq if ch != ","])
        index = [str(i) for i in range(seq_len)]

        # 정답 라벨 인덱스 모음
        positive_label = set(get_bs(bs))  # helpers.py의 get_bs 사용
        negative_label = set(index) - positive_label

        # 예측 라벨 인덱스 모음
        positive_pred = set(bps.split(",")) if (bps is not None and bps != "") else set()
        negative_pred = set(index) - positive_pred

        TP = len(positive_pred & positive_label)
        TN = len(negative_pred & negative_label)
        FP = len(positive_pred & negative_label)
        FN = len(negative_pred & positive_label)

        T_TP += TP
        T_TN += TN
        T_FP += FP
        T_FN += FN

    precision   = T_TP / (T_TP + T_FP + eps)
    recall      = T_TP / (T_TP + T_FN + eps)
    specificity = T_TN / (T_TN + T_FP + eps)
    ACC         = (T_TP + T_TN) / (T_TP + T_TN + T_FP + T_FN + eps)
    G_mean      = np.sqrt(max(specificity, 0.0) * max(recall, 0.0))
    F1          = (2 * precision * recall) / (precision + recall + eps)
    F2          = (5 * precision * recall) / (4 * precision + recall + eps)

    return (np.round(precision, 3), np.round(recall, 3), np.round(specificity, 3),
            np.round(ACC, 3), np.round(G_mean, 3), np.round(F1, 3), np.round(F2, 3))

def fwrite(path, IDs, pred_binding_sites=None, binding_sites=None,threshold=0.6):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(IDs) != len(pred_binding_sites):
        print(f"[WARN] len(IDs)={len(IDs)} vs len(pred)={len(pred_binding_sites)} 불일치. "
              "zip 때문에 뒤쪽 행이 잘릴 수 있습니다.")

    def _to_str(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) > 0 and isinstance(x[0], (float, np.floating)):
                idx = [i for i, p in enumerate(x) if p >= threshold]
                return ",".join(map(str, idx))
            return ",".join(map(str, x))
        return str(x)

    pred_strs = [_to_str(p) for p in pred_binding_sites]

    written = 0
    with open(path, "w") as fw:
        if binding_sites is not None:
            fw.write("PDB\tBS\tPred_BS\n")
            for id_, bs, pbs in zip(IDs, binding_sites, pred_strs):
                bs_str = "" if bs is None else str(bs)
                fw.write(f"{id_}\t{bs_str}\t{pbs}\n")
                written += 1
        else:
            fw.write("PDB\tPred_BS\n")
            for id_, pbs in zip(IDs, pred_strs):
                fw.write(f"{id_}\t{pbs}\n")
                written += 1

    print(f"[INFO] Wrote {written} rows to {path}")
    if written == 0:
        print("[WARN] 한 줄도 쓰이지 않았습니다. pred/ID 길이 또는 예측 내용 확인 필요.")

def input_check(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError(f"{path} does not exist.")
    return path

def main():
    parser = argparse.ArgumentParser(
        description="Planet-X predicts binding site based on protein sequence information"
    )
    parser.add_argument("--config", "-c", required=True, type=input_check,
                        help="The file contains information on the protein sequences to predict binding sites.")
    parser.add_argument("--labels", dest="labels", action="store_true",
                        help="Include binding site labels (training/evaluation).")
    parser.add_argument("--no-labels", dest="labels", action="store_false",
                        help="Exclude labels (test/inference).")
    parser.set_defaults(labels=True)
    args = parser.parse_args()

    config = load_cfg(args.config)

    eval_cfg = config.get("eval", {})
    th = float(eval_cfg.get("threshold", 0.6))

    print("1. Load data ...")
    if args.labels:
        with open(config["paths"]["prot_feats"], "rb") as f:
            IDs, sequences, binding_sites, protein_feats = pickle.load(f)
    else:
        with open(config["paths"]["prot_feats"], "rb") as f:
            IDs, sequences, protein_feats = pickle.load(f)
        binding_sites = None 

    print("2. Make dataset ...")
    dataset = PocketDataset(IDs, protein_feats, sequences)
    loader = Dataloader(dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    print("3. Binding sites prediction ...")
    trainiter = PlanetXTrainIter(config)

    ckpt_root = config["paths"]["save_path"] 
    fold_ckpts = sorted(glob.glob(os.path.join(ckpt_root, "fold*", "Planet_X.pth")))
    if not fold_ckpts:
        raise FileNotFoundError(f"No Planet_X.pth found under {ckpt_root}/fold*/")
    print(f"  Found {len(fold_ckpts)} folds: {fold_ckpts}")

    predicted_binding_sites = trainiter.run_test(loader, fold_ckpts)

    predicted_binding_sites_str = convert_bs(predicted_binding_sites, threshold=th)

    if args.labels:
        non_empty = sum(1 for s in predicted_binding_sites_str if s)
        print(f"[DEBUG] non-empty predictions: {non_empty}/{len(predicted_binding_sites_str)}")
        print(f"[DEBUG] sample pred[0]: {predicted_binding_sites_str[0] if predicted_binding_sites_str else None}")

        precision, recall, specificity, ACC, G_mean, F1, F2 = get_results(
            binding_sites, predicted_binding_sites_str, sequences
        )
        print(f"[RESULT] Precision={precision}, Recall={recall}, "
              f"Specificity={specificity}, ACC={ACC}, G_mean={G_mean}, "
              f"F1={F1}, F2={F2}")

    print("4. Write predicted binding sites ...")
    if args.labels:
        fwrite(config["paths"]["result_path"], IDs,
               pred_binding_sites=predicted_binding_sites_str,
               binding_sites=binding_sites)
    else:
        fwrite(config["paths"]["result_path"], IDs,
               pred_binding_sites=predicted_binding_sites_str)

if __name__ == "__main__":
    main()
