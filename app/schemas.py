# app/schemas.py
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime, time

class UserCreate(BaseModel):
    user_id: UUID
    name: str
    email: str
    age: int
    weight: float
    created_at: datetime

class UsersTargetWakingPeriodCreate(BaseModel):
    user_id: UUID
    target_start_time: str  # 使用 ISO 格式的時間字串
    target_end_time: str    # 使用 ISO 格式的時間字串

class UsersRealSleepDataCreate(BaseModel):
    user_id: UUID
    start_time: datetime
    end_time: datetime

class UsersRealTimeIntakeCreate(BaseModel):
    user_id: UUID
    drink_name: str
    caffeine_amount: int
    taking_timestamp: datetime

class UsersPVTResultsCreate(BaseModel):
    user_id: UUID
    mean_rt: float
    lapses: int
    false_starts: int
    test_at: datetime
    device: str

class RecommendationsCaffeineCreate(BaseModel):
    user_id: UUID
    recommended_caffeine_amount: int
    recommended_caffeine_intake_timing: time
    
class AlertnessDataCreate(BaseModel):
    user_id: UUID
    timestamp: datetime
    awake: bool
    g_PD: float
    P0_values: float
    P_t_caffeine: float
    P_t_no_caffeine: float
    P_t_real: float