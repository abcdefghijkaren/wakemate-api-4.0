# app/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base

# 使用 Render 提供的 External Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://wakemate_user:neNuQ9GMsK7lahBvAsM5b9atg9ijsrwI@dpg-d1pm85ruibrs73dt6bpg-a.oregon-postgres.render.com/user_info_wakemate_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
