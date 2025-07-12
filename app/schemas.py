# app/schemas.py
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

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
    takeing_timestamp: datetime

class UsersPVTResultsCreate(BaseModel):
    user_id: UUID
    mean_rt: float
    lapses: int
    false_starts: int
    test_at: datetime
    device: str
