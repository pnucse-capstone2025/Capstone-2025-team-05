# esm_embed.py
import sys
import os
sys.path.append(os.path.abspath("../binding_site"))

import torch
import esm
import numpy as np

# ---------------------------
# 1. ESM2 모델 및 변환기 초기화
# ---------------------------
print("[INFO] Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
batch_converter = alphabet.get_batch_converter()
print("[INFO] ESM-2 model ready.")

# ---------------------------
# 2. 단백질 시퀀스 → 임베딩 함수
# ---------------------------
def embed_sequence(sequence: str) -> np.ndarray:
    """
    ESM-2 모델을 사용하여 단일 시퀀스를 임베딩합니다.
    
    Args:
        sequence (str): 단백질 시퀀스. 다중 체인일 경우 ','로 구분.

    Returns:
        np.ndarray: shape (L_total, 1280), float32
    """
    chain_seqs = [s.strip() for s in sequence.split(",") if s.strip()]
    all_embeddings = []

    with torch.no_grad():
        for chain in chain_seqs:
            data = [("protein", chain)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            out = model(tokens, repr_layers=[33], return_contacts=False)
            token_repr = out["representations"][33]
            emb = token_repr[0, 1:-1]  # [CLS], [EOS] 제거
            emb_np = emb.detach().cpu().numpy().astype(np.float32)

            all_embeddings.extend(emb_np)

    return np.array(all_embeddings, dtype=np.float32)