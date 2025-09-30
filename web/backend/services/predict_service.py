import requests
import json
import hashlib
import traceback
from flask import jsonify, request, Response, stream_with_context
from extensions import db
from models.prediction import Prediction
from models.structure import Structure
from flask_jwt_extended import get_jwt_identity
from datetime import datetime


# GPU 서버 주소
GPU_SERVER_URL = "http://planx.ngrok.io"


# -------------------------------
# 1) 예측 시작 (POST /predict/start)
# -------------------------------
def predict_start():
    try:
        data = request.get_json()

        sequence = data.get("sequence")
        smiles = data.get("smiles")
        protname = data.get("protname")

        if not sequence or not smiles or not protname:
            return jsonify({"error": "All fields (sequence, smiles, protname) are required."}), 400

        # 로그인한 사용자 ID 
        user_id = get_jwt_identity()

        # prot_id = 내부용 해시
        sequence_hash = hashlib.sha1(sequence.encode()).hexdigest()
        internal_id = "prot_" + sequence_hash[:10]

        # DB에 Prediction 엔트리 먼저 생성 (pending)
        prediction = Prediction(
            user_id=user_id,
            protname=protname,
            sequence=sequence,
            smiles=smiles,
            status="pending",
            created_at=datetime.utcnow()
        )
        db.session.add(prediction)
        db.session.commit()

        return jsonify({"prediction_id": prediction.id}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -------------------------------
# 2) SSE 스트리밍 (GET /predict/stream/<prediction_id>)
# -------------------------------
def predict_stream(prediction_id):
    try:
        # DB에서 prediction 불러오기
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            return jsonify({"error": "Invalid prediction_id"}), 404

        sequence_hash = hashlib.sha1(prediction.sequence.encode()).hexdigest()
        internal_id = "prot_" + sequence_hash[:10]

        payload = {
            "id": internal_id,
            "sequence": prediction.sequence,
            "smiles": prediction.smiles
        }

        def generate():
            with requests.post(f"{GPU_SERVER_URL}/predict_all", json=payload, stream=True) as resp:
                resp.raise_for_status()

                binding_sites = []
                affinity_score = None
                pdb_url = None

                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data:"):
                        continue

                    event = json.loads(line.decode("utf-8").replace("data: ", ""))
                    stage = event.get("stage")

                    if stage == "quick":
                        # 빠른 결과 저장
                        binding_sites = event.get("binding_sites", [])
                        affinity_score = event.get("affinity", None)

                        prediction.binding_sites = json.dumps(binding_sites)
                        prediction.binding_affinity = affinity_score
                        prediction.status = "running"
                        db.session.commit()

                        event["prediction_id"] = prediction_id
                        yield f"data: {json.dumps(event)}\n\n"

                    elif stage == "final":
                        pdb_url = event.get("pdb_url")

                        if pdb_url:
                            existing_structure = Structure.query.filter_by(
                                sequence_hash=sequence_hash,
                                structure_url=pdb_url
                            ).first()

                            if not existing_structure:
                                structure = Structure(
                                    sequence_hash=sequence_hash,
                                    structure_url=pdb_url,
                                    created_at=datetime.utcnow()
                                )
                                db.session.add(structure)
                                db.session.commit()
                            else:
                                structure = existing_structure

                            prediction.structure_id = structure.id
                            prediction.status = "done"
                            db.session.commit()

                        event["prediction_id"] = prediction_id
                        yield f"data: {json.dumps(event)}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except requests.exceptions.RequestException as e:
        prediction = Prediction.query.get(prediction_id)
        if prediction:
            prediction.status = "failed"
            prediction.error_message = str(e)
            db.session.commit()
        return jsonify({"error": "GPU 서버 요청 실패", "details": str(e)}), 502

    except Exception as e:
        prediction = Prediction.query.get(prediction_id)
        if prediction:
            prediction.status = "failed"
            prediction.error_message = str(e)
            db.session.commit()
        return jsonify({"error": "예측 중 오류 발생", "details": str(e)}), 500
