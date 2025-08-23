# app/schemas.py
from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime, time
from typing import List, Optional

# --- 使用者註冊 ---
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str  # 新增

class UserResponse(BaseModel):
    user_id: UUID
    name: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True  # 替代 orm_mode = True

# --- 使用者設定清醒區間 ---
class UsersTargetWakingPeriodCreate(BaseModel):
    user_id: UUID
    target_start_time: datetime  # ISO 格式時間字串
    target_end_time: datetime

# --- 使用者實際睡眠資料 ---
class UsersRealSleepDataCreate(BaseModel):
    user_id: UUID
    sleep_start_time: datetime
    sleep_end_time: datetime

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


# --- Device ------------------------------------------------
from typing import Optional

# ========== Heart Rate ==========
class DeviceHeartRateBase(BaseModel):
    time: datetime
    heartrate: float
    confidence: float
    source: Optional[str] = None
    user_id: UUID

class DeviceHeartRateDataCreate(DeviceHeartRateBase):
    time: datetime
    heartrate: int
    confidence: int
    source: Optional[str] = None  # 可選
    user_id: Optional[UUID] = None  # 可選

class DeviceHeartRateResponse(DeviceHeartRateBase):
    id: int
    saved_at: datetime

    class Config:
        from_attributes = True

# ========== XYZ Time ==========
class DeviceXYZTimeBase(BaseModel):
    timestamp: str
    x: str
    y: str
    z: str
    user_id: Optional[UUID] = None

class DeviceXYZTimeDataCreate(DeviceXYZTimeBase):
    timestamp: str
    x: str
    y: str
    z: str
    user_id: Optional[UUID] = None
    
class DeviceXYZTimeResponse(DeviceXYZTimeBase):
    id: int
    saved_at: datetime

    class Config:
        from_attributes = True

# 批量用 schema------------------------------------
class BulkHeartRate(BaseModel):
    records: List[DeviceHeartRateDataCreate]


class BulkXYZTime(BaseModel):
    records: List[DeviceXYZTimeDataCreate]