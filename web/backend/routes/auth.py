from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

from models.user import User
from models.verified_email import VerifiedEmail
from models.verification_code import VerificationCode
from extensions import db
from services.email_service import send_verification_email
from datetime import timedelta

auth_bp = Blueprint("auth", __name__)

# -------------------------------
# 1. 인증 코드 이메일 발송
# -------------------------------
@auth_bp.route("/auth/send-code", methods=["POST"])
def send_code():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"error": "이메일이 제공되지 않았습니다."}), 400
    
    if User.query.filter_by(email=email, is_deleted=False).first():
        return jsonify({"error": "이미 가입된 이메일입니다."}), 409

    try:
        code = send_verification_email(email)
        return jsonify({"message": "인증 코드가 이메일로 전송되었습니다."})  
    except Exception as e:
        return jsonify({"error": "이메일 전송에 실패했습니다.", "details": str(e)}), 500


# -------------------------------
# 2. 인증 코드 확인
# -------------------------------
@auth_bp.route("/auth/verify-code", methods=["POST"])
def verify_code():
    data = request.get_json()
    email = data.get("email")
    code = data.get("code")

    if not email or not code:
        return jsonify({"error": "이메일과 인증 코드 모두 필요합니다."}), 400

    vc = VerificationCode.query.filter_by(email=email).first()
    if not vc:
        return jsonify({"error": "인증 코드가 존재하지 않습니다. 먼저 코드를 요청하세요."}), 404

    if vc.code != code:
        return jsonify({"error": "인증 코드가 일치하지 않습니다."}), 401

    if vc.is_expired():
        db.session.delete(vc)
        db.session.commit()
        return jsonify({"error": "인증 코드가 만료되었습니다. 다시 요청해주세요."}), 410

    if not VerifiedEmail.query.filter_by(email=email).first():
        db.session.add(VerifiedEmail(email=email))

    db.session.delete(vc) 
    db.session.commit()

    return jsonify({"message": "이메일 인증 완료!"})


# -------------------------------
# 3. 회원가입
# -------------------------------
@auth_bp.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    nickname = data.get("nickname")

    # 필수 항목 확인
    if not email or not password or not nickname:
        return jsonify({"error": "이메일, 비밀번호, 닉네임 모두 필요합니다."}), 400

    # 이메일 인증 확인
    verified = VerifiedEmail.query.filter_by(email=email).first()
    if not verified:
        return jsonify({"error": "이메일 인증이 필요합니다."}), 400

    # 이메일 중복 확인
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "이미 가입된 이메일입니다."}), 409

    # 닉네임 중복 확인
    if User.query.filter_by(nickname=nickname).first():
        return jsonify({"error": "이미 사용 중인 닉네임입니다."}), 409

    # 비밀번호 해시화
    hashed_password = generate_password_hash(password)

    # 사용자 생성 및 DB 저장
    new_user = User(email=email, password=hashed_password, nickname=nickname)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "회원가입이 완료되었습니다!"}), 201
# -------------------------------
# 4. 로그인
# -------------------------------
@auth_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email, is_deleted=False).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "이메일 또는 비밀번호가 올바르지 않습니다."}), 401

    access_token = create_access_token(identity=str(user.id), expires_delta=timedelta(hours=6))

    return jsonify({"token": access_token, "nickname": user.nickname})


# -------------------------------
# 5. 회원 탈퇴
# -------------------------------
@auth_bp.route("/auth/delete", methods=["DELETE"])
@jwt_required()
def delete_account():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    if not user or user.is_deleted:
        return jsonify({"error": "존재하지 않거나 이미 탈퇴한 계정입니다."}), 404

    user.is_deleted = True

    VerifiedEmail.query.filter_by(email=user.email).delete()
    VerificationCode.query.filter_by(email=user.email).delete()

    db.session.commit()

    return jsonify({"message": "회원 탈퇴가 완료되었습니다."})
