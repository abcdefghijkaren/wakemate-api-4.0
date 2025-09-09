# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from app import models, schemas
from app.models import (
    User,
    UsersTargetWakingPeriod,
    UsersRealSleepData,
    UsersRealTimeIntake,
    UsersPVTResults,
    RecommendationsCaffeine,
    AlertnessDataForVisualization,
    DeviceHeartRateData,
    DeviceXYZTimeData,
)
from app.schemas import (
    UserCreate,
    UsersTargetWakingPeriodCreate,
    UsersRealSleepDataCreate,
    UsersRealTimeIntakeCreate,
    UsersPVTResultsCreate,
    AlertnessDataCreate,
    UserResponse,
    DeviceHeartRateDataCreate,
    DeviceXYZTimeDataCreate,
)
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4, UUID
from passlib.context import CryptContext
from .database import engine, SessionLocal
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import logging

# 計算模組
from core.caffeine_recommendation import run_caffeine_recommendation
from core.alertness_data import run_alertness_data
from core.database import get_db_connection

# ---------------- 日誌設定 ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- 初始化 ----------------
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


# ------- 封裝共用觸發邏輯 -------
def trigger_calculation(user_id: UUID):
    """執行咖啡因與清醒度計算"""
    try:
        logger.info(f"開始觸發運算：user_id={user_id}")
        conn = get_db_connection()
        run_caffeine_recommendation(conn, user_id)
        run_alertness_data(conn, user_id)
        conn.close()
        logger.info(f"✅ 運算完成：user_id={user_id}")
        return {"status": "ok", "message": f"calculation finished for {user_id}"}
    except Exception as e:
        logger.error(f"❌ 運算失敗：user_id={user_id}, error={str(e)}")
        return {"status": "error", "message": str(e)}


# Scheduler 每小時補算
def scheduled_job():
    conn = get_db_connection()
    try:
        logger.info("執行排程運算（全部使用者）")
        run_caffeine_recommendation(conn)
        run_alertness_data(conn)
        logger.info("✅ 排程運算完成")
    except Exception as e:
        logger.error(f"❌ 排程運算失敗：{str(e)}")
    finally:
        conn.close()


scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_job, "interval", hours=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())


@app.get("/ping")
def ping(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "success", "message": "連線成功 ✅"}
    except Exception as e:
        logger.error(f"資料庫連線錯誤: {str(e)}")
        return {"status": "error", "message": str(e)}


# ========== API新增資料 ==========
@app.post("/users/", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = pwd_context.hash(user.password)
        new_user = User(
            user_id=uuid4(), email=user.email, hashed_password=hashed_password, name=user.name
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except Exception as e:
        db.rollback()
        logger.error(f"❌ 註冊失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"註冊失敗: {str(e)}")


@app.post("/users_sleep/")
def create_user_sleep(data: schemas.UsersRealSleepDataCreate, db: Session = Depends(get_db)):
    try:
        entry = models.UsersRealSleepData(**data.dict())
        db.add(entry)
        db.commit()
        db.refresh(entry)

        calc_result = trigger_calculation(entry.user_id)
        return {"status": "success", "id": entry.id, "calculation": calc_result}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ 新增睡眠資料失敗: {str(e)}")
        raise HTTPException(status_code=400, detail=f"新增睡眠資料失敗: {str(e)}")


@app.post("/users_wake/")
def create_user_wake(data: schemas.UsersTargetWakingPeriodCreate, db: Session = Depends(get_db)):
    try:
        entry = models.UsersTargetWakingPeriod(**data.dict())
        db.add(entry)
        db.commit()
        db.refresh(entry)

        calc_result = trigger_calculation(entry.user_id)
        return {"status": "success", "id": entry.id, "calculation": calc_result}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ 新增清醒目標失敗: {str(e)}")
        raise HTTPException(status_code=400, detail=f"新增清醒目標失敗: {str(e)}")


@app.post("/users_intake/")
def create_user_intake(data: schemas.UsersRealTimeIntakeCreate, db: Session = Depends(get_db)):
    try:
        entry = models.UsersRealTimeIntake(**data.dict())
        db.add(entry)
        db.commit()
        db.refresh(entry)

        calc_result = trigger_calculation(entry.user_id)
        return {"status": "success", "id": entry.id, "calculation": calc_result}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ 新增咖啡因攝取資料失敗: {str(e)}")
        raise HTTPException(status_code=400, detail=f"新增咖啡因攝取資料失敗: {str(e)}")


@app.post("/users_pvt/")
def create_user_pvt(data: schemas.UsersPVTResultsCreate, db: Session = Depends(get_db)):
    try:
        entry = models.UsersPVTResults(**data.dict())
        db.add(entry)
        db.commit()
        db.refresh(entry)

        calc_result = trigger_calculation(entry.user_id)
        return {"status": "success", "id": entry.id, "calculation": calc_result}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ 新增PVT資料失敗: {str(e)}")
        raise HTTPException(status_code=400, detail=f"新增PVT資料失敗: {str(e)}")