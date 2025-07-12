# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Time, Float
from sqlalchemy.dialects.postgresql import UUID
from .database import Base
import uuid

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    name = Column(String)
    email = Column(String)
    weight = Column(Numeric)
    age = Column(Integer)
    created_at = Column(DateTime(timezone=True))

class UsersTargetWakingPeriod(Base):
    __tablename__ = "users_target_waking_period"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True))
    target_start_time = Column(Time(timezone=True))
    target_end_time = Column(Time(timezone=True))

class UsersRealSleepData(Base):
    __tablename__ = "users_real_sleep_data"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True))
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))

class UsersRealTimeIntake(Base):
    __tablename__ = "users_real_time_intake"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True))
    drink_name = Column(String)
    caffeine_amount = Column(Integer)
    takeing_timestamp = Column(DateTime(timezone=True))

class UsersPVTResults(Base):
    __tablename__ = "users_pvt_results"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True))
    mean_rt = Column(Float)
    lapses = Column(Integer)
    false_starts = Column(Integer)
    test_at = Column(DateTime(timezone=True))
    device = Column(String)

class RecommendationsCaffeine(Base):
    __tablename__ = "recommendations_caffeine"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True))
    recommended_caffeine_amount = Column(Integer)
    recommended_caffeine_intake_timing = Column(Time(timezone=True))
