from extensions import db
from datetime import datetime

class Structure(db.Model):
    __tablename__ = 'structures'

    # 기본 키 (자동 증가)
    id = db.Column(db.Integer, primary_key=True)
    
    # 단백질 서열의 해시값 (SHA1/MD5 등) → 동일 서열 중복 방지 및 빠른 조회를 위해 인덱스 설정
    sequence_hash = db.Column(db.String(64), nullable=False, index=True) 

    # 구조 파일 저장 경로 
    structure_url = db.Column(db.Text, nullable=False) 

    # 레코드 생성 시간 (자동 저장)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
