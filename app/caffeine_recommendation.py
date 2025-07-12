import pandas as pd
import numpy as np
import psycopg2  # 假設使用 PostgreSQL
from psycopg2.extras import execute_values
from datetime import datetime, timedelta

# 連接到資料庫
conn = psycopg2.connect(
    dbname='user_info_wakemate_db',
    user='postgres',
    password='123456',
    host='localhost',
    port='5432'
)
cursor = conn.cursor()

# 從資料庫讀取數據
cursor.execute("SELECT user_id, target_start_time, target_end_time FROM users_target_waking_period")
waking_periods = cursor.fetchall()

cursor.execute("SELECT user_id, start_time, end_time FROM users_real_sleep_data")
sleep_data = cursor.fetchall()

# 參數設定
M_c = 1.1
k_a = 1.0
k_c = 0.5
P0_base = 270
dose_unit = 100
max_daily_dose = 300

# 模擬咖啡因影響與攝取建議
recommendations = []

for user_id, target_start_time, target_end_time in waking_periods:
    # 這裡假設每個用戶的睡眠數據都是連續的
    for user_id_sleep, start_time, end_time in sleep_data:
        if user_id == user_id_sleep:
            sleep_hour = start_time.hour
            wake_hour = end_time.hour
            total_hours = 24  # 假設計算一天
            t = np.arange(0, total_hours + 1)

            # Sigmoid 函數模擬每日清醒度變化
            def sigmoid(x, L=100, x0=14, k=0.2):
                return L / (1 + np.exp(-k * (x - x0)))

            # 計算每小時基礎清醒度
            P0_values = np.zeros_like(t, dtype=float)
            awake_flags = np.ones_like(t, dtype=bool)

            for h in range(24):
                if sleep_hour < wake_hour:
                    asleep = (h >= sleep_hour and h < wake_hour)
                else:
                    asleep = (h >= sleep_hour or h < wake_hour)
                awake_flags[h] = not asleep
                P0_values[h] = P0_base + sigmoid(h) if not asleep else P0_base

            # 模擬咖啡因影響與攝取建議
            g_PD = np.ones_like(t, dtype=float)
            P_t_caffeine = np.copy(P0_values)
            intake_schedule = []

            daily_dose = 0
            for hour in range(24):
                if not awake_flags[hour]:
                    continue
                if P_t_caffeine[hour] > 270:
                    if daily_dose + dose_unit <= max_daily_dose:
                        intake_schedule.append((user_id, hour, dose_unit))
                        daily_dose += dose_unit
                        t_0 = hour
                        effect = 1 / (1 + (M_c * dose_unit / 200) * (k_a / (k_a - k_c)) *
                                      (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                        effect = np.where(t < t_0, 1, effect)
                        g_PD *= effect
                        P_t_caffeine = P0_values * g_PD

            # 儲存建議結果
            for user_id, hour, dose in intake_schedule:
                # 將小時轉換為完整的時間戳格式
                recommended_time = datetime.combine(datetime.today(), datetime.min.time()) + timedelta(hours=hour)
                recommendations.append((user_id, dose, recommended_time.strftime('%Y-%m-%d %H:%M:%S')))

# 將建議結果寫入資料庫
execute_values(cursor, "INSERT INTO recommendations_caffeine (user_id, recommended_caffeine_amount, recommended_caffeine_intake_timing) VALUES %s", recommendations)

# 提交變更並關閉連接
conn.commit()
cursor.close()
conn.close()

