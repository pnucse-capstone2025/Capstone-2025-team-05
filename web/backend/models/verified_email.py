from extensions import db  
from datetime import datetime

class VerifiedEmail(db.Model):
    __tablename__ = 'verified_emails'

    # 기본 키 (자동 증가)
    id = db.Column(db.Integer, primary_key=True)

    # 인증 완료된 이메일
    email = db.Column(db.String(120), unique=True, nullable=False)

    # 인증 완료 시각 (기본값: 생성 시 현재 시간)
    verified_at = db.Column(db.DateTime, default=datetime.utcnow)