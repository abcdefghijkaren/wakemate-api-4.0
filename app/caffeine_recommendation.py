import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import os

# === 連接資料庫 ===
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT", "5432")
)
cursor = conn.cursor()

# === 讀取資料 ===
cursor.execute("SELECT user_id, target_start_time, target_end_time FROM users_target_waking_period")
waking_periods = cursor.fetchall()

cursor.execute("SELECT user_id, start_time, end_time FROM users_real_sleep_data")
sleep_data = cursor.fetchall()

# === 預設參數（若沒有個人化參數時使用平均值） ===
DEFAULT_PARAMS = {"M_c": 1.1, "k_a": 1.0, "k_c": 0.5}
P0_base = 270
max_daily_dose = 400   # mg
dose_min, dose_max = 50, 200  # 每次建議劑量上下限

# Sigmoid 函數
def sigmoid(x, L=100, x0=14, k=0.2):
    return L / (1 + np.exp(-k * (x - x0)))

# 咖啡因效應公式
def caffeine_effect(t, t0, dose, M_c, k_a, k_c):
    effect = 1 / (1 + (M_c * dose / 200) * (k_a / (k_a - k_c)) *
                  (np.exp(-k_c * (t - t0)) - np.exp(-k_a * (t - t0))))
    return np.where(t < t0, 1.0, effect)

recommendations = []

for user_id, target_start_time, target_end_time in waking_periods:
    # 取得個人化參數（若存在資料表 user_params）
    cursor.execute("""
        SELECT M_c, k_a, k_c
        FROM user_params
        WHERE user_id = %s
    """, (user_id,))
    row = cursor.fetchone()
    if row:
        M_c, k_a, k_c = row
    else:
        M_c, k_a, k_c = DEFAULT_PARAMS.values()

    # 對應使用者的睡眠
    for user_id_sleep, start_time, end_time in sleep_data:
        if user_id != user_id_sleep:
            continue

        sleep_hour = start_time.hour
        wake_hour = end_time.hour
        total_hours = 24
        t = np.arange(0, total_hours + 1)

        # === 計算 P0 (baseline) ===
        P0_values = np.zeros_like(t, dtype=float)
        awake_flags = np.ones_like(t, dtype=bool)

        for h in range(24):
            if sleep_hour < wake_hour:
                asleep = (h >= sleep_hour and h < wake_hour)
            else:
                asleep = (h >= sleep_hour or h < wake_hour)
            awake_flags[h] = not asleep
            P0_values[h] = P0_base + sigmoid(h) if not asleep else P0_base

        # === 模擬推薦攝取 ===
        g_PD = np.ones_like(t, dtype=float)
        P_t_caffeine = np.copy(P0_values)
        intake_schedule = []
        daily_dose = 0

        for hour in range(24):
            if not awake_flags[hour]:
                continue

            if P_t_caffeine[hour] > 270:  # 超過門檻才需要咖啡因
                # 缺口 ΔP
                delta = P_t_caffeine[hour] - 270

                # 粗略反推所需劑量 (線性近似，可再優化)
                dose_needed = min(max(dose_min, delta), dose_max)

                if daily_dose + dose_needed <= max_daily_dose:
                    intake_schedule.append((user_id, hour, dose_needed))
                    daily_dose += dose_needed

                    t0 = hour
                    effect = caffeine_effect(t, t0, dose_needed, M_c, k_a, k_c)
                    g_PD *= effect
                    P_t_caffeine = P0_values * g_PD

        # === 儲存建議結果 ===
        for user_id, hour, dose in intake_schedule:
            recommended_time = datetime.combine(datetime.today(), datetime.min.time()) + timedelta(hours=hour)
            recommendations.append((user_id, dose, recommended_time.strftime('%Y-%m-%d %H:%M:%S')))

# 寫入資料庫
if recommendations:
    execute_values(cursor, """
        INSERT INTO recommendations_caffeine 
        (user_id, recommended_caffeine_amount, recommended_caffeine_intake_timing)
        VALUES %s
    """, recommendations)

conn.commit()
cursor.close()
conn.close()