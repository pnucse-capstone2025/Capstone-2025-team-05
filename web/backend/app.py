from flask import Flask
from flask_cors import CORS
from flask_mail import Mail
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from flask_migrate import Migrate
import os

# 커스텀 확장 및 블루프린트
from extensions import db, mail
from models.verified_email import VerifiedEmail  
from routes.upload import upload_bp
from routes.predict import predict_bp
from routes.auth import auth_bp

from create_tables import create_all_tables


# -----------------------
# 1. 환경변수 로딩
# -----------------------
load_dotenv()

# -----------------------
# 2. Flask 앱 생성 및 CORS 허용
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# 3. 앱 환경설정
# -----------------------

## 데이터베이스 설정
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

## 시크릿 키 설정
app.secret_key = os.getenv("SECRET_KEY")

## JWT 설정
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY")
jwt = JWTManager(app)

## 이메일 설정
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

# -----------------------
# 4. 확장 초기화
# -----------------------
db.init_app(app)
mail.init_app(app)

migrate = Migrate(app, db)

# -----------------------
# 5. 블루프린트 등록
# -----------------------
app.register_blueprint(upload_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(auth_bp)

# -----------------------
# 6. 기본 라우트
# -----------------------
@app.route("/", methods=["GET"])
def root():
    return {"message": "Flask 연결 성공"}

# -----------------------
# 7. 서버 실행
# -----------------------
if __name__ == "__main__":

    #create_all_tables(app)

    app.run(host="0.0.0.0", port=5001)
