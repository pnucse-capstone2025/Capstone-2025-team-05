from extensions import db
from models.user import User
from models.structure import Structure
from datetime import datetime
from sqlalchemy.sql import func

class Prediction(db.Model):
    __tablename__ = 'predictions'

    # 기본 키 (자동 증가)
    id = db.Column(db.Integer, primary_key=True)

    # 사용자 정보
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

    # 사용자 지정 단백질 닉네임 (ex: "myprotein1")
    protname = db.Column(db.String(100), nullable=False)

    # 입력 데이터
    sequence = db.Column(db.Text, nullable=False)
    smiles = db.Column(db.Text, nullable=False)

    # 예측 결과
    binding_sites = db.Column(db.Text, nullable=False, default="[]")  # JSON 문자열 (예: "[4, 12, 17]")
    binding_affinity = db.Column(db.Float, nullable=True)  # 예: 7.52 (pKd)

    # 구조 참조 (선택적)
    structure_id = db.Column(db.Integer, db.ForeignKey('structures.id'), nullable=True)
    structure = db.relationship('Structure', backref=db.backref('predictions', lazy=True))

    # 상태 (pending / running / done / failed)
    status = db.Column(db.String(20), default="pending", nullable=False)

    # 실패 원인 메시지 (optional)
    error_message = db.Column(db.Text, nullable=True)
    
    # 요청 시각
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    # 아카이빙 플래그
    archived = db.Column(db.Boolean, default=False)

    # 썸네일 PNG (S3 URL)
    screenshot_url = db.Column(db.Text, nullable=True) 