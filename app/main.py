# app/main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import app.models as models
from app.models import User, UsersTargetWakingPeriod, UsersRealSleepData, UsersRealTimeIntake, UsersPVTResults, RecommendationsCaffeine, AlertnessDataForVisualization
from app.schemas import UserCreate, UsersTargetWakingPeriodCreate, UsersRealSleepDataCreate, UsersRealTimeIntakeCreate, UsersPVTResultsCreate, AlertnessDataCreate
from .database import SessionLocal
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()

# 依賴注入：取得 DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# cors中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

@app.get("/ping")
def ping(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "success", "message": "連線成功 ✅"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ========== 取得資料 ==========
@app.get("/users/")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.get("/users_sleep/")
def get_sleep_data(db: Session = Depends(get_db)):
    return db.query(UsersRealSleepData).all()

@app.get("/users_intake/")
def get_intake_data(db: Session = Depends(get_db)):
    return db.query(UsersRealTimeIntake).all()

@app.get("/users_pvt/")
def get_pvt_results(db: Session = Depends(get_db)):
    return db.query(UsersPVTResults).all()

@app.get("/users_wake/")
def get_wake_target(db: Session = Depends(get_db)):
    return db.query(UsersTargetWakingPeriod).all()

@app.get("/recommendations/")
def get_recommendations(db: Session = Depends(get_db)):
    return db.query(RecommendationsCaffeine).all()

@app.get("/alertness_data/")
def get_alertness_data(db: Session = Depends(get_db)):
    return db.query(AlertnessDataForVisualization).all()

# ========== 新增資料 ==========
@app.post("/users/")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # 自動產生 UUID（假設你的 User model 中 user_id 是 UUID 欄位）
    db_user = User(
        user_id=str(uuid.uuid4()),  # 自動產生 user_id
        **user.dict()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"status": "success", "user_id": db_user.user_id}

@app.post("/users_wake/")
def create_user_wake(data: UsersTargetWakingPeriodCreate, db: Session = Depends(get_db)):
    entry = UsersTargetWakingPeriod(**data.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "id": entry.id}

@app.post("/users_sleep/")
def create_user_sleep(data: UsersRealSleepDataCreate, db: Session = Depends(get_db)):
    entry = UsersRealSleepData(**data.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "id": entry.id}

@app.post("/users_intake/")
def create_user_intake(data: UsersRealTimeIntakeCreate, db: Session = Depends(get_db)):
    entry = UsersRealTimeIntake(**data.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "id": entry.id}

@app.post("/users_pvt/")
def create_user_pvt(data: UsersPVTResultsCreate, db: Session = Depends(get_db)):
    entry = UsersPVTResults(**data.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "id": entry.id}

@app.post("/alertness_data/")
def create_alertness_data(data: AlertnessDataCreate, db: Session = Depends(get_db)):
    entry = AlertnessDataForVisualization(**data.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "id": entry.id}