# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app import models, schemas
import app.models as models
from app.models import User, UsersTargetWakingPeriod, UsersRealSleepData, UsersRealTimeIntake, UsersPVTResults, RecommendationsCaffeine, AlertnessDataForVisualization, DeviceHeartRateData
from app.schemas import UserCreate, UsersTargetWakingPeriodCreate, UsersRealSleepDataCreate, UsersRealTimeIntakeCreate, UsersPVTResultsCreate, AlertnessDataCreate, UserResponse, DeviceHeartRateDataCreate
from .database import SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from passlib.context import CryptContext
from .database import engine

models.Base.metadata.create_all(bind=engine)
app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

@app.get("/device_heart_rate/")
def get_device_heart_rate_data(db: Session = Depends(get_db)):
    return db.query(DeviceHeartRateData).all()

# ========== 新增資料 ==========
@app.post("/users/", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # 檢查 email 是否已存在
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = pwd_context.hash(user.password)
    new_user = User(
        user_id=uuid4(),
        email=user.email,
        hashed_password=hashed_password,
        name=user.name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user  # 自動轉換為 UserResponse

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

@app.post("/device_heart_rate/")
def create_device_heart_rate(data: DeviceHeartRateDataCreate, db: Session = Depends(get_db)):
    entry = DeviceHeartRateData(**data.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "id": entry.id}
