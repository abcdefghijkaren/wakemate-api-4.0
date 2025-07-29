# app/models.py
from sqlalchemy import (
    Column, Integer, String, DateTime, Numeric, Time,
    Float, Boolean, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    weight = Column(Numeric, nullable=True)
    age = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships (optional but helpful)
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

    user = relationship("User", back_populates="waking_periods")


class UsersRealSleepData(Base):
    __tablename__ = "users_real_sleep_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)

    user = relationship("User", back_populates="sleep_data")


class UsersRealTimeIntake(Base):
    __tablename__ = "users_real_time_intake"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    drink_name = Column(String, nullable=False)
    caffeine_amount = Column(Integer, nullable=False)
    taking_timestamp = Column(DateTime(timezone=True), nullable=False)

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

    user = relationship("User", back_populates="pvt_results")


class RecommendationsCaffeine(Base):
    __tablename__ = "recommendations_caffeine"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    recommended_caffeine_amount = Column(Integer, nullable=False)
    recommended_caffeine_intake_timing = Column(Time(timezone=True), nullable=False)

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

    user = relationship("User", back_populates="alertness_data")