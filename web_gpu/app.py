# app.py
import os
import sys
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

sys.path.insert(0, "../binding_site")


# ESM 임베딩, AlphaFold 구조 예측, Binding site / Affinity 모델 불러오기
from esm_embed import embed_sequence
from structure_service import run_alphafold
from predict_binding_site import predict_binding_sites
from predict_binding_affinity import predict as predict_affinity_model

load_dotenv()

app = Flask(__name__)
CORS(app)

# ESM-2 모델의 최대 서열 길이 제한 (안정성 확보)
MAX_SEQUENCE_LENGTH = 1500

# -------------------------
# 1. 단백질 구조 예측 API
# -------------------------
@app.route("/predict_structure", methods=["POST"])
def predict_structure():
    """
    입력: { "id": str, "sequence": str }
    처리: ColabFold 실행 → PDB 파일 생성 및 S3 업로드
    출력: { "id": str, "pdb_url": str }
    """
    try:
        data = request.get_json()
        print("📩 /predict_structure 요청 수신:", data)

        # 요청 형식 검증
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. JSON body is required."}), 400

        prot_id = data.get("id")
        sequence = data.get("sequence")

        if not (prot_id and sequence):
            return jsonify({"error": "Missing 'id' or 'sequence'."}), 400

        # 서열 전처리 및 길이 제한 확인
        cleaned_sequence = sequence.replace(",", "").strip()
        if len(cleaned_sequence) > MAX_SEQUENCE_LENGTH:
            return jsonify({"error": f"Sequence too long. Maximum {MAX_SEQUENCE_LENGTH} residues."}), 413

        # AlphaFold 실행
        print(f"🚀 run_alphafold 호출 시작: prot_id={prot_id}, 길이={len(cleaned_sequence)}")
        pdb_url = run_alphafold(cleaned_sequence, prot_id)
        print("✅ run_alphafold 결과:", pdb_url)

        return jsonify({"id": prot_id, "pdb_url": pdb_url})

    except Exception as e:
        print("❌ /predict_structure 오류:", str(e))
        return jsonify({"error": str(e)}), 500

# -------------------------
# 2. 바인딩 사이트 예측 API
# -------------------------
@app.route("/predict_binding_site", methods=["POST"])
def predict_binding_site():
    """
    입력: { "id": str, "sequence": str }
    처리: ESM 임베딩 → PlanetX 모델로 binding site residue 예측
    출력: { "id": str, "binding_sites": list[int] }
    """
    try:
        data = request.get_json()
        print("📩 /predict_binding_site 요청 수신:", data)

        prot_id = data.get("id")
        sequence = data.get("sequence")

        if not (prot_id and sequence):
            return jsonify({"error": "Missing 'id' or 'sequence'."}), 400

        # 서열 길이 확인
        cleaned_sequence = sequence.replace(",", "").strip()
        if len(cleaned_sequence) > MAX_SEQUENCE_LENGTH:
            return jsonify({"error": f"Sequence too long. Maximum {MAX_SEQUENCE_LENGTH} residues."}), 413

        # 1) 단백질 임베딩 (ESM-2)
        print("🔍 ESM 임베딩 시작")
        embedding = embed_sequence(sequence)
        print("✅ ESM 임베딩 완료")

        # 2) Binding site 예측
        print("🔍 Binding site 예측 시작")
        binding_sites = predict_binding_sites(prot_id, sequence, embedding)
        print("✅ Binding site 예측 완료:", binding_sites[:10], "...")

        return jsonify({"id": prot_id, "binding_sites": binding_sites})

    except Exception as e:
        print("❌ /predict_binding_site 오류:", str(e))
        return jsonify({"error": str(e)}), 500

# -------------------------
# 3. 바인딩 친화도 예측 API
# -------------------------
@app.route("/predict_affinity", methods=["POST"])
def predict_affinity():
    """
    입력: { "sequence": str, "smiles": str }
    처리: ProteinLigandModel (ESM + ChemBERTa)로 binding affinity 예측
    출력: { "affinity": float }
    """
    try:
        data = request.get_json()
        print("📩 /predict_affinity 요청 수신:", data)

        seq = data.get("sequence")
        smiles = data.get("smiles")

        if not (seq and smiles):
            return jsonify({"error": "Missing 'sequence' or 'smiles'."}), 400

        # Affinity 예측
        print("🔍 Affinity 예측 시작")
        affinity_score = predict_affinity_model(seq, smiles)
        print("✅ Affinity 예측 완료:", affinity_score)

        return jsonify({"affinity": affinity_score})

    except Exception as e:
        print("❌ /predict_affinity 오류:", str(e))
        return jsonify({"error": str(e)}), 500

# -------------------------
# 4. 통합 예측 API (SSE 스트리밍)
# -------------------------
@app.route("/predict_all", methods=["POST"])
def predict_all():
    """
    서버-발송 이벤트(Server-Sent Events) 기반 통합 API:
    1단계 (quick): binding site + affinity 예측 결과를 먼저 반환
    2단계 (final): AlphaFold 구조 예측 후 PDB URL까지 반환
    """
    try:
        data = request.get_json()
        print("📩 /predict_all 요청 수신:", data)

        prot_id = data.get("id")
        sequence = data.get("sequence")
        smiles = data.get("smiles")

        if not (prot_id and sequence and smiles):
            return jsonify({"error": "Missing 'id', 'sequence', or 'smiles'."}), 400
            
        # 길이 제한 체크
        cleaned_sequence = sequence.replace(",", "").strip()
        if len(cleaned_sequence) > MAX_SEQUENCE_LENGTH:
            return jsonify({"error": f"Sequence too long. Maximum {MAX_SEQUENCE_LENGTH} residues."}), 413

        # SSE generator 함수 정의
        def generate():
            # 1. 빠른 예측 (ESM → binding site / affinity)
            print("🔍 Quick 결과 생성 시작")
            embedding = embed_sequence(sequence)
            binding_sites = predict_binding_sites(prot_id, sequence, embedding)
            affinity_score = predict_affinity_model(sequence, smiles)
            print("✅ Quick 결과 생성 완료")

            quick_result = {
                "stage": "quick",
                "id": prot_id,
                "sequence": sequence,
                "binding_sites": binding_sites,
                "affinity": affinity_score,
                "pdb_url": None
            }
            yield f"data: {json.dumps(quick_result)}\n\n"

            # 2. AlphaFold 구조 예측 (시간 오래 걸림)
            print("🚀 run_alphafold 호출 시작 (최종 단계)")
            pdb_url = run_alphafold(cleaned_sequence, prot_id)
            print("✅ run_alphafold 완료:", pdb_url)

            final_result = {
                "stage": "final",
                "id": prot_id,
                "sequence": sequence,
                "binding_sites": binding_sites,
                "affinity": affinity_score,
                "pdb_url": pdb_url
            }
            yield f"data: {json.dumps(final_result)}\n\n"
            
        # SSE 응답 반환
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        print("❌ /predict_all 오류:", str(e))
        return jsonify({"error": str(e)}), 500

# -------------------------
# 5. Flask 서버 실행
# -------------------------
if __name__ == "__main__":
    print("🚀 GPU 서버 Flask 시작")
    app.run(host="0.0.0.0", port=5050, debug=False)