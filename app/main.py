# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from app import models, schemas
from app.models import User, UsersTargetWakingPeriod, UsersRealSleepData, UsersRealTimeIntake, UsersPVTResults, RecommendationsCaffeine, AlertnessDataForVisualization, DeviceHeartRateData, DeviceXYZTimeData
from app.schemas import UserCreate, UsersTargetWakingPeriodCreate, UsersRealSleepDataCreate, UsersRealTimeIntakeCreate, UsersPVTResultsCreate, AlertnessDataCreate, UserResponse, DeviceHeartRateDataCreate, DeviceXYZTimeDataCreate
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4, UUID
from passlib.context import CryptContext
from .database import engine, SessionLocal

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

@app.post("/users_sleep/")
def create_user_sleep(data: UsersRealSleepDataCreate, db: Session = Depends(get_db)):
    entry = UsersRealSleepData(**data.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "success", "id": entry.id}

@app.post("/users_wake/")
def create_user_wake(data: UsersTargetWakingPeriodCreate, db: Session = Depends(get_db)):
    entry = UsersTargetWakingPeriod(**data.dict())
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

# @app.post("/device_heart_rate/")
# def create_device_heart_rate(data: DeviceHeartRateDataCreate, db: Session = Depends(get_db)):
#     entry = DeviceHeartRateData(**data.dict())
#     db.add(entry)
#     db.commit()
#     db.refresh(entry)
#     return {"status": "success", "id": entry.id}

# @app.post("/device_xyz_time/")
# def create_device_xyz_time(data: DeviceXYZTimeDataCreate, db: Session = Depends(get_db)):
#     entry = DeviceXYZTimeData(**data.dict())
#     db.add(entry)
#     db.commit()
#     db.refresh(entry)
#     return {"status": "success", "id": entry.id}


# ========== 取得資料 ==========
@app.get("/users/")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.get("/users_sleep/")
def get_sleep_data(user_id: UUID = Query(None), db: Session = Depends(get_db)):
    query = db.query(UsersRealSleepData)
    if user_id:
        query = query.filter(UsersRealSleepData.user_id == user_id)
    return query.all()

@app.get("/users_wake/")
def get_wake_target(user_id: UUID = Query(None), db: Session = Depends(get_db)):
    query = db.query(UsersTargetWakingPeriod)
    if user_id:
        query = query.filter(UsersTargetWakingPeriod.user_id == user_id)
    return query.all()

@app.get("/users_intake/")
def get_intake_data(user_id: UUID = Query(None), db: Session = Depends(get_db)):
    query = db.query(UsersRealTimeIntake)
    if user_id:
        query = query.filter(UsersRealTimeIntake.user_id == user_id)
    return query.all()

@app.get("/users_pvt/")
def get_pvt_results(user_id: UUID = Query(None), db: Session = Depends(get_db)):
    query = db.query(UsersPVTResults)
    if user_id:
        query = query.filter(UsersPVTResults.user_id == user_id)
    return query.all()

@app.get("/recommendations/")
def get_recommendations(user_id: UUID = Query(None), db: Session = Depends(get_db)):
    query = db.query(RecommendationsCaffeine)
    if user_id:
        query = query.filter(RecommendationsCaffeine.user_id == user_id)
    return query.all()

@app.get("/alertness_data/")
def get_alertness_data(user_id: UUID = Query(None), db: Session = Depends(get_db)):
    query = db.query(AlertnessDataForVisualization)
    if user_id:
        query = query.filter(AlertnessDataForVisualization.user_id == user_id)
    return query.all()

# @app.get("/device_heart_rate/")
# def get_device_heart_rate_data(db: Session = Depends(get_db)):
#     return db.query(DeviceHeartRateData).all()

# @app.get("/device_xyz_time/")
# def get_device_xyz_time(db: Session = Depends(get_db)):
#     return db.query(DeviceXYZTimeData).all()


# ====================== 以下是 DEVICE 的資料批量傳送端口====================================

# ================== 批量寫入 Heart Rate ==================
@app.post("/device/heart_rate/bulk", response_model=list[schemas.DeviceHeartRateResponse])
def create_heart_rate_bulk(payload: schemas.BulkHeartRate, db: Session = Depends(get_db)):
    objs = [models.DeviceHeartRate(**record.dict()) for record in payload.records]
    db.add_all(objs)
    db.commit()
    for obj in objs:
        db.refresh(obj)
    return objs


@app.get("/device/heart_rate", response_model=list[schemas.DeviceHeartRateResponse])
def get_heart_rate(db: Session = Depends(get_db)):
    return db.query(models.DeviceHeartRate).all()


# ================== 批量寫入 XYZ Time ==================
@app.post("/device/xyz_time/bulk", response_model=list[schemas.DeviceXYZTimeResponse])
def create_xyz_time_bulk(payload: schemas.BulkXYZTime, db: Session = Depends(get_db)):
    try:
        objs = [models.DeviceXYZTimeData(**record.dict()) for record in payload.records]
        db.add_all(objs)
        db.commit()
        for obj in objs:
            db.refresh(obj)
        return objs
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"寫入失敗: {str(e)}")


@app.get("/device/xyz_time", response_model=list[schemas.DeviceXYZTimeResponse])
def get_xyz_time(db: Session = Depends(get_db)):
    return db.query(models.DeviceXYZTimeData).all()