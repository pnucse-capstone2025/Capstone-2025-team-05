# predict_binding_site.py
import os
import numpy as np
import sys
import glob

sys.path.append("../binding_site")

from modules.utils import load_cfg
from modules.data import PocketDataset, Dataloader
from modules.TrainIters import PlanetXTrainIter
from test import convert_bs

# -------------------------
# 1. 설정 로드
# -------------------------
CONFIG_PATH = "../binding_site/configuration.yml"
config = load_cfg(CONFIG_PATH)

# 2. 5개 fold 학습된 모델 가중치 경로 전부 가져오기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 web 폴더 기준
ckpt_root = os.path.normpath(os.path.join(BASE_DIR, "../binding_site", config["paths"]["save_path"]))
fold_ckpts = sorted(glob.glob(os.path.join(ckpt_root, "fold*", "Planet_X.pth")))  # run_test는 리스트 받으니까 그대로 리스트로

if not fold_ckpts:
    raise FileNotFoundError(f"No Planet_X.pth found under {ckpt_root}/fold*/")
    
print(f"[INFO] Found {len(fold_ckpts)} folds: {fold_ckpts}")

# -------------------------
# 3. PlanetX 모델 초기화 (프로세스 내에서 1회만)
# -------------------------
print("[INFO] Loading PlanetX model...")
train_iter = PlanetXTrainIter(config)
print("[INFO] PlanetX model ready.")

# -------------------------
# 4. 바인딩 사이트 예측 함수
# -------------------------
def predict_binding_sites(prot_id: str, sequence: str, embedding: np.ndarray) -> list:
    """
    단백질 ID, 서열, ESM 임베딩을 입력으로 받아
    Planet_X 모델을 통해 바인딩 사이트 residue index를 예측한다.
    """
    # (1) 입력을 PocketDataset → Dataloader 형태로 변환
    dataset = PocketDataset([prot_id], [embedding], [sequence])
    loader = Dataloader(dataset, batch_size=1, shuffle=False)

    # (2) 여러 fold 모델로 inference 수행 후 soft voting
    #     -> fold별 확률 점수를 평균해 더 안정적인 결과 도출
    pred_scores = train_iter.run_test(loader, fold_ckpts)

    # (3) 확률 점수를 threshold=0.55로 이진화하여 바인딩 사이트 residue index 추출
    pred_indices = convert_bs(pred_scores, threshold=0.55)

    # (4) 문자열 "3,7,8" 형태를 → [3, 7, 8] 리스트로 변환
    return list(map(int, pred_indices[0].split(","))) if pred_indices[0] else []