# structure_service.py
import os
import subprocess
import boto3
from botocore.exceptions import ClientError  # S3 캐시 확인 시 예외 처리용
from dotenv import load_dotenv

load_dotenv()

# 환경변수 불러오기
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# 전역 S3 클라이언트 생성
s3 = boto3.client("s3", region_name=AWS_REGION)

def upload_to_s3(file_path: str, protein_id: str) -> str:
    """
    PDB 파일을 S3에 업로드하고 URL 반환
    """
    key = f"structures/{protein_id}/rank_001.pdb"
    s3.upload_file(
        Filename=file_path,
        Bucket=S3_BUCKET_NAME,
        Key=key,
        ExtraArgs={"ACL": "public-read", "ContentType": "application/octet-stream"}
    )
    # 퍼블릭 URL 직접 구성
    url = f"https://s3.{AWS_REGION}.amazonaws.com/{S3_BUCKET_NAME}/{key}"
    return url



def run_alphafold(sequence: str, protein_id: str, save_dir="outputs/structure") -> str:
    """
    1. 캐시 확인: S3에 이미 구조가 있으면 바로 URL 반환
    2. 없으면 ColabFold 실행:
       - 입력 FASTA 생성
       - ColabFold batch 실행
       - rank_001.pdb 선택
    3. S3 업로드 후 URL 반환
    """
    print(f"[START] run_alphafold: protein_id={protein_id}, seq_len={len(sequence)}")

    # ✅ 0. S3 캐시 확인 (이미 있으면 ColabFold 실행 생략)
    key = f"structures/{protein_id}/rank_001.pdb"
    try:
        s3.head_object(Bucket=S3_BUCKET_NAME, Key=key) # 존재 여부 확인
        url = f"https://s3.{AWS_REGION}.amazonaws.com/{S3_BUCKET_NAME}/{key}"
        print(f"[INFO] S3 캐시 히트: {url}")
        return url
    except ClientError:
        print(f"[INFO] 캐시 없음 → ColabFold 실행 시작")

    os.makedirs(save_dir, exist_ok=True)

    # 1. 입력 FASTA 파일 저장
    fasta_path = os.path.join(save_dir, f"{protein_id}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{protein_id}\n{sequence.strip()}")
    print(f"[INFO] FASTA 저장 완료: {fasta_path}")

    # 2. ColabFold 실행
    output_path = os.path.join(save_dir, protein_id)
    os.makedirs(output_path, exist_ok=True)

    print(f"[INFO] ColabFold 실행 시작... output_dir={output_path}")
    try:
        result = subprocess.run(
            [
                "colabfold_batch",
                "--msa-mode", "single_sequence",        # 다중서열정렬(MSA) 대신 단일 서열만 사용
                "--num-recycle", "1",                   # 구조 refinement 반복 횟수 최소화
                fasta_path,
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",   # GPU 메모리 선점 방지
                "XLA_PYTHON_CLIENT_MEM_FRACTION": ".3",     # GPU 메모리 사용 제한
                "CUDA_VISIBLE_DEVICES": "2",                # 특정 GPU만 사용
            },
        )
        print("[INFO] ColabFold STDOUT:\n", result.stdout)
        if result.stderr:
            print("[WARN] ColabFold STDERR:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        # 실행 실패 시 stdout/stderr 로그 출력
        print("[ERROR] ColabFold 실행 실패!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError("AlphaFold prediction failed.")

    # 3. ColabFold 출력 중 rank_001.pdb 선택 (가장 좋은 모델)
    best_pdb = None
    for fname in os.listdir(output_path):
        if "rank_001" in fname and fname.endswith(".pdb"):
            best_pdb = os.path.join(output_path, fname)
            break

    if not best_pdb:
        print(f"[ERROR] rank_001 PDB 파일을 찾을 수 없음. output_path={output_path}")
        raise FileNotFoundError("rank_001 PDB not found in ColabFold output.")

    print(f"[INFO] Best PDB 선택됨: {best_pdb}")

    # 4. 최종 PDB를 S3 업로드 후 URL 반환
    url = upload_to_s3(best_pdb, protein_id)
    print(f"[DONE] run_alphafold 완료: {url}")
    return url