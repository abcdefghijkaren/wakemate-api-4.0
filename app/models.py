# app/models.py
from sqlalchemy import (
    Column, Integer, String, DateTime, Numeric, Time,
    Float, Boolean, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base
import uuid
from sqlalchemy.sql import func
from datetime import datetime, timezone

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    waking_periods = relationship("UsersTargetWakingPeriod", back_populates="user")
    sleep_data = relationship("UsersRealSleepData", back_populates="user")
    intake_data = relationship("UsersRealTimeIntake", back_populates="user")
    pvt_results = relationship("UsersPVTResults", back_populates="user")
    recommendations = relationship("RecommendationsCaffeine", back_populates="user")
    alertness_data = relationship("AlertnessDataForVisualization", back_populates="user")


class UsersTargetWakingPeriod(Base):
    __tablename__ = "users_target_waking_period"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    target_start_time = Column(Time(timezone=True), nullable=False)
    target_end_time = Column(Time(timezone=True), nullable=False)
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="waking_periods")


class UsersRealSleepData(Base):
    __tablename__ = "users_real_sleep_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    sleep_start_time = Column(DateTime(timezone=True), nullable=False)
    sleep_end_time = Column(DateTime(timezone=True), nullable=False)
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="sleep_data")


class UsersRealTimeIntake(Base):
    __tablename__ = "users_real_time_intake"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    drink_name = Column(String, nullable=False)
    caffeine_amount = Column(Integer, nullable=False)
    taking_timestamp = Column(DateTime(timezone=True), nullable=False)
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="intake_data")


class UsersPVTResults(Base):
    __tablename__ = "users_pvt_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    mean_rt = Column(Float, nullable=False)
    lapses = Column(Integer, nullable=False)
    false_starts = Column(Integer, nullable=False)
    test_at = Column(DateTime(timezone=True), nullable=False)
    device = Column(String, nullable=True)
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="pvt_results")


class RecommendationsCaffeine(Base):
    __tablename__ = "recommendations_caffeine"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    recommended_caffeine_amount = Column(Integer, nullable=False)
    recommended_caffeine_intake_timing = Column(Time(timezone=True), nullable=False)
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="recommendations")


class AlertnessDataForVisualization(Base):
    __tablename__ = "alertness_data_for_visualization"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    awake = Column(Boolean, nullable=False)
    g_PD = Column(Float, nullable=False)
    P0_values = Column(Float, nullable=False)
    P_t_caffeine = Column(Float, nullable=False)
    P_t_no_caffeine = Column(Float, nullable=False)
    P_t_real = Column(Float, nullable=False)
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="alertness_data")


class DeviceHeartRateData(Base):
    __tablename__ = "device_heart_rate_data"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(DateTime(timezone=True), nullable=False)
    heartrate = Column(Integer, nullable=False)
    confidence = Column(Integer, nullable=False)
    source = Column(String, nullable=True)  # 允許為空
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)  # 允許為空
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # user = relationship("User ", back_populates="device_heart_rate_data")  # 暫時不連結


class DeviceXYZTimeData(Base):
    __tablename__ = "device_xyz_time"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Time(timezone=False), nullable=False)  # 只有時間，無日期
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)  # 允許為空
    saved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
