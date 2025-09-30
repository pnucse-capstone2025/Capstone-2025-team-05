# backend/routes/predict.py
import os
import json
import boto3
import uuid
from flask import Blueprint, request, Response, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, decode_token

from services.predict_service import predict_start, predict_stream
from extensions import db
from models.prediction import Prediction

predict_bp = Blueprint("predict", __name__)

# --- S3 설정 ---
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
s3 = boto3.client("s3", region_name=AWS_REGION)

# 1. 예측 시작 (DB에 Prediction 생성만)
@predict_bp.route("/predict/start", methods=["POST"])
@jwt_required()
def predict_start_route():
    return predict_start()


# 2. SSE 스트리밍 (GPU 서버 호출 + 결과 중계)
@predict_bp.route("/predict/stream/<int:prediction_id>", methods=["GET"])
def predict_stream_route(prediction_id):
    token = request.args.get("token")
    if not token:
        return jsonify({"error": "Missing token"}), 401
    try:
        decoded = decode_token(token)
        user_id = int(decoded["sub"])  
    except Exception as e:
        return jsonify({"error": "Invalid token", "details": str(e)}), 401

    return predict_stream(prediction_id)


# 3) Archive 플래그 설정
@predict_bp.route("/predictions/<int:prediction_id>/archive", methods=["POST"])
@jwt_required()
def archive_prediction(prediction_id):
    try:
        user_id = int(get_jwt_identity()) 
        pred = db.session.get(Prediction, prediction_id)
        if not pred:
            return jsonify({"error": "Prediction not found"}), 404
        if pred.user_id != user_id:
            return jsonify({"error": "Unauthorized"}), 403

        pred.archived = True
        db.session.commit()
        return jsonify({
            "message": "Prediction archived successfully",
            "id": prediction_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 4) Archived 목록
@predict_bp.route("/predictions/archived", methods=["GET"])
@jwt_required()
def list_archived_predictions():
    try:
        user_id = int(get_jwt_identity())
        preds = Prediction.query.filter_by(user_id=user_id, archived=True).all()
        return jsonify([
            {
                "id": p.id,
                "protname": p.protname,
                "sequence": p.sequence,
                "affinity": p.binding_affinity,
                "binding_sites": json.loads(p.binding_sites or "[]"),
                "pdb_url": (p.structure.structure_url if p.structure else None),
                "screenshot_url": p.screenshot_url,
                "created_at": (p.created_at.isoformat() + "Z" if p.created_at else None),
            }
            for p in preds
        ]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
# 5) 스크린샷 업로드 (Save 누를 때 캡처한 PNG 업로드)
@predict_bp.route("/upload/screenshot/<int:prediction_id>", methods=["POST"])
@jwt_required()
def upload_screenshot(prediction_id):
    try:
        user_id = int(get_jwt_identity())
        pred = db.session.get(Prediction, prediction_id)
        if not pred:
            return jsonify({"error": "Prediction not found"}), 404
        if pred.user_id != user_id:
            return jsonify({"error": "Unauthorized"}), 403

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        key = f"screenshots/{prediction_id}_{uuid.uuid4().hex}.png"
        s3.upload_fileobj(
            file,
            S3_BUCKET_NAME,
            key,
            ExtraArgs={"ACL": "public-read", "ContentType": "image/png"}
        )
        url = f"https://s3.{AWS_REGION}.amazonaws.com/{S3_BUCKET_NAME}/{key}"
        
        pred.screenshot_url = url
        db.session.commit()

        return jsonify({"url": url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
