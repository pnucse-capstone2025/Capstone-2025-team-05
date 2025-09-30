# predict_binding_affinity.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GPU 선택

import sys
# binding_affinity 폴더를 모듈 경로에 추가
sys.path.append("/workspace/Capstone-2025-team-05/binding_affinity")

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors

from planet_bap import ProteinLigandModel
from esm_embed import embed_sequence   # 사전에 로드된 ESM-2 모델 활용 (embedding 함수)


# ----------------------------
# 1. 환경 설정
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ChemBERTa 모델과 학습된 ProteinLigandModel 경로 (학습 시 설정과 동일해야 함)
chembert_model_name = "seyonec/ChemBERTa-zinc-base-v1"
model_path = "/workspace/binding_affinity/best2020/best_model.pt"

# ----------------------------
# 2. 모델 로드
# ----------------------------
print("Loading ChemBERTa...")
chembert_model = AutoModel.from_pretrained(chembert_model_name)
chembert_tokenizer = AutoTokenizer.from_pretrained(chembert_model_name)

print("Loading trained ProteinLigandModel...")
model = ProteinLigandModel(chembert_model=chembert_model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# 3. Protein featurization (ESM 임베딩 + 물리화학 feature)
# ----------------------------
# 아미노산 잔기별 charge/polar/hydrophobicity/분자량 정보 사전
charge_dict = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 1}
polar_dict  = {"S":1,"T":1,"N":1,"Q":1,"Y":1,"C":1,"D":1,"E":1,"K":1,"R":1,"H":1}
hydro_dict  = {
    "A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,
    "K":-3.9,"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,
    "T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3
}
weight_dict = {
    "A":89.1,"C":121.2,"D":133.1,"E":147.1,"F":165.2,"G":75.1,"H":155.2,"I":131.2,
    "K":146.2,"L":131.2,"M":149.2,"N":132.1,"P":115.1,"Q":146.2,"R":174.2,
    "S":105.1,"T":119.1,"V":117.1,"W":204.2,"Y":181.2
}

def clean_sequence(seq: str) -> str:
    """허용되지 않은 문자 → 'X'로 치환 (ESM 입력 안정화용)"""
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZ")
    return "".join([aa if aa in allowed else "X" for aa in seq])

def get_physchem_features(sequence: str) -> np.ndarray:
    """각 residue별 물리화학적 feature 추출"""
    feats = []
    for aa in sequence:
        charge = charge_dict.get(aa, 0)
        acidic  = 1 if charge == -1 else 0
        basic   = 1 if charge == 1 else 0
        neutral = 1 if charge == 0 else 0
        polar   = polar_dict.get(aa, 0)
        hydro   = hydro_dict.get(aa, 0.0)
        weight  = weight_dict.get(aa, 0.0)
        feats.append([acidic, basic, neutral, polar, hydro, weight])
    return np.array(feats, dtype=np.float32)

def featurize_protein(seq: str) -> torch.Tensor:
    """
    단백질 서열 → (ESM-2 임베딩 + residue-level 물리화학 feature) 텐서 반환
    최종 shape: (L, 1286)   [1280 (ESM) + 6 (physchem)]
    """
    seq = clean_sequence(seq)

    # ESM-2 임베딩 (L,1280)
    emb = embed_sequence(seq)   # (L,1280) numpy

    # 물리화학 feature (L,6)
    phys = get_physchem_features(seq)  # (L,6) numpy

    # 두 feature를 concat
    combined = np.concatenate([phys, emb], axis=1)  # (L,1286)

    return torch.from_numpy(combined).float()

# ----------------------------
# 4. Ligand featurization
# ----------------------------
def featurize_ligand(smiles: str, max_len=128):
    encoded = chembert_tokenizer(
        smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len
    )
    smi_ids = encoded["input_ids"].squeeze(0)
    smi_mask = encoded["attention_mask"].squeeze(0).bool()
    return smi_ids, smi_mask

def compute_global_properties(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * 10         # SMILES 파싱 실패 시 0으로 채움
    return [
        Descriptors.MolWt(mol),                 # 분자량
        Descriptors.MolLogP(mol),               # 지용성 
        Descriptors.TPSA(mol),                  # 극성 표면적
        Descriptors.NumRotatableBonds(mol),     # 회전 가능한 결합 수
        Descriptors.HeavyAtomCount(mol),        # heavy atom 개수
        Descriptors.FractionCSP3(mol),          # sp3 탄소 비율
        Descriptors.NumHDonors(mol),            # 수소 공여자
        Descriptors.NumHAcceptors(mol),         # 수소 수용자
        Descriptors.RingCount(mol),             # 고리 개수
        Descriptors.MolMR(mol),                 # 분자 굴절률
    ]

# ----------------------------
# 5. Prediction 함수
# ----------------------------
def predict(seq: str, smiles: str):
    """
    입력: 단백질 서열, 리간드 SMILES
    출력: 예측된 binding affinity (float)
    """
    # protein feature (1,L,1286)
    pt_feat = featurize_protein(seq).unsqueeze(0).to(device)   # (1,L,1286)

    # ligand feature (ChemBERTa tokenization)
    smi_ids, smi_mask = featurize_ligand(smiles)
    smi_ids = smi_ids.unsqueeze(0).to(device)                  # (1,S)
    smi_mask = smi_mask.unsqueeze(0).to(device)                # (1,S)

    # ligand 전역 물성 (1,10)
    lig_global = torch.tensor(compute_global_properties(smiles)).unsqueeze(0).float().to(device)  

    # 모델 추론
    with torch.no_grad():
        out = model(pt_feat, smi_ids, smi_mask, lig_global)    # (1,1)
        return out.view(-1).cpu().item()

# ----------------------------
# 6. 실행 테스트
# ----------------------------
if __name__ == "__main__":
    test_seq = "MKTFFVVALAAAGALA,GQEVLIRLFKSHPETL"  # 단백질 서열 (실제 입력으로 교체)
    test_smiles = "CNC1=NC2=C(CC[NH2+]CC3CCCC3)C3=C(C=C2N1)C(=O)N[C](N)N3"    # 실제 입력 시(smiles)
    pred = predict(test_seq, test_smiles)
    print(f"예측된 Binding Affinity: {pred:.4f}")