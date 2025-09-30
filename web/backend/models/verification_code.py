from extensions import db
from datetime import datetime, timedelta

class VerificationCode(db.Model):
    __tablename__ = 'verification_codes'

    # 기본 키 (자동 증가)
    id = db.Column(db.Integer, primary_key=True)

    # 인증 대상 이메일
    email = db.Column(db.String(120), unique=True, nullable=False)

    # 인증 코드 (6자리 숫자)
    code = db.Column(db.String(6), nullable=False)

    # 코드 생성 시각
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 만료 시각
    expires_at = db.Column(db.DateTime, nullable=False)

    def is_expired(self):
        # 만료 여부 확인
        return datetime.utcnow() > self.expires_at