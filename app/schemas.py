# app/schemas.py
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime, time

# --- 使用者註冊 ---
class UserCreate(BaseModel):
    name: str
    email: str
    age: int
    weight: float
    created_at: datetime

class UserResponse(BaseModel):
    user_id: UUID
    name: str
    email: str
    age: int
    weight: float
    created_at: datetime

    class Config:
        orm_mode = True

# --- 使用者設定清醒區間 ---
class UsersTargetWakingPeriodCreate(BaseModel):
    user_id: UUID
    target_start_time: str  # ISO 格式時間字串
    target_end_time: str

# --- 使用者實際睡眠資料 ---
class UsersRealSleepDataCreate(BaseModel):
    user_id: UUID
    start_time: datetime
    end_time: datetime

# --- 使用者即時咖啡因攝取資料 ---
class UsersRealTimeIntakeCreate(BaseModel):
    user_id: UUID
    drink_name: str
    caffeine_amount: int
    taking_timestamp: datetime

# --- 使用者反應時間測試結果 (PVT) ---
class UsersPVTResultsCreate(BaseModel):
    user_id: UUID
    mean_rt: float
    lapses: int
    false_starts: int
    test_at: datetime
    device: str

# --- 建議攝取資訊 ---
class RecommendationsCaffeineCreate(BaseModel):
    user_id: UUID
    recommended_caffeine_amount: int
    recommended_caffeine_intake_timing: time

# --- 清醒度資料 ---
class AlertnessDataCreate(BaseModel):
    user_id: UUID
    timestamp: datetime
    awake: bool
    g_PD: float
    P0_values: float
    P_t_caffeine: float
    P_t_no_caffeine: float
    P_t_real: float