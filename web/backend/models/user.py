from extensions import db  
from datetime import datetime

class User(db.Model):
    __tablename__ = 'users'

    # 기본 키 (자동 증가)
    id = db.Column(db.Integer, primary_key=True)

    # 사용자 이메일 (로그인 ID로 사용, 고유 값)
    email = db.Column(db.String(120), unique=True, nullable=False)

    # 비밀번호
    password = db.Column(db.String(200), nullable=False)  

    # 사용자 닉네임
    nickname = db.Column(db.String(50), nullable=False)

    # 계정 생성 시각
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 마지막 업데이트 시각
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 계정 삭제 여부 (soft delete 플래그)
    is_deleted = db.Column(db.Boolean, default=False)