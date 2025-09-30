import random
import string
from flask_mail import Message
from datetime import datetime, timedelta
from extensions import mail, db
from models.verification_code import VerificationCode


def generate_verification_code(length=6):
    """
    랜덤 인증 코드 생성 (기본 6자리 숫자)
    """
    return ''.join(random.choices(string.digits, k=length))


def send_verification_email(email):
    """
    인증 코드 생성 후 해당 이메일로 발송하고 DB에 저장 (또는 갱신)
    """
    code = generate_verification_code()
    expires_at = datetime.utcnow() + timedelta(minutes=10)

    # 기존 코드가 있으면 갱신, 없으면 새로 추가
    existing = VerificationCode.query.filter_by(email=email).first()
    if existing:
        existing.code = code
        existing.created_at = datetime.utcnow()
        existing.expires_at = expires_at
    else:
        new_code = VerificationCode(email=email, code=code, expires_at=expires_at)
        db.session.add(new_code)

    db.session.commit()

    # 이메일 전송
    msg = Message(
        subject="[team05] 이메일 인증 코드",
        recipients=[email],
        body=f"안녕하세요!\n\n인증 코드: {code}\n\n해당 코드를 입력하여 인증을 완료해주세요."
    )

    mail.send(msg)
    return code
