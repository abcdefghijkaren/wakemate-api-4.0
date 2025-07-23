import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import psycopg2  # 假設使用 PostgreSQL 作為資料庫
from psycopg2.extras import execute_values
import os

# ===== 連接到資料庫 =====
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT", "5432")
)
cursor = conn.cursor()

# ===== 從資料庫讀取資料 =====
# 讀取咖啡因攝取數據
caffeine_query = "SELECT * FROM users_real_time_intake;"
caffeine_df = pd.read_sql_query(caffeine_query, conn, parse_dates=["taking_timestamp"])

# 讀取睡眠數據
sleep_query = "SELECT * FROM users_real_sleep_data;"
sleep_df = pd.read_sql_query(sleep_query, conn, parse_dates=["start_time", "end_time"])

# 讀取目標清醒時間
target_query = "SELECT * FROM users_target_waking_period;"
target_df = pd.read_sql_query(target_query, conn, parse_dates=["target_start_time", "target_end_time"])

# ===== 模型參數設定 =====
M_c = 1.1
k_a = 1.0
k_c = 0.5
P0_base = 270

def sigmoid(x, L=100, x0=14, k=0.2):
    return L / (1 + np.exp(-k * (x - x0)))

# ===== 建立時間軸 =====
start_time = min(caffeine_df["taking_timestamp"].min(),
                 sleep_df["start_time"].min()).replace(minute=0, second=0)
end_time = max(caffeine_df["taking_timestamp"].max(),
               sleep_df["end_time"].max()).replace(minute=0, second=0) + timedelta(hours=1)
total_hours = int((end_time - start_time).total_seconds() // 3600)
time_index = [start_time + timedelta(hours=i) for i in range(total_hours + 1)]
t = np.arange(total_hours + 1)

# ===== 標記清醒與睡眠狀態，並建立 P0 =====
awake_flags = np.ones(len(time_index), dtype=bool)
P0_values = np.zeros(len(time_index), dtype=float)

for i, time in enumerate(time_index):
    is_awake = True
    for _, row in sleep_df.iterrows():
        if row["start_time"] <= time < row["end_time"]:
            is_awake = False
            break
    awake_flags[i] = is_awake
    hour = time.hour
    P0_values[i] = P0_base + sigmoid(hour) if is_awake else P0_base

# ===== 計算咖啡因效應 (caffeine_log) =====
g_PD = np.ones(len(time_index), dtype=float)

for _, row in caffeine_df.iterrows():
    take_time = row["taking_timestamp"]
    dose = float(row["caffeine_amount"])
    t_0 = int((take_time - start_time).total_seconds() // 3600)
    if t_0 >= len(t):
        continue

    effect = 1 / (1 + (M_c * dose / 200) * (k_a / (k_a - k_c)) *
                  (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
    effect = np.where(t < t_0, 1, effect)
    g_PD *= effect

P_t_caffeine = P0_values * g_PD

# ===== 新增計算：實時攝取量的清醒度 =====
g_PD_real = np.ones(len(time_index), dtype=float)

for _, row in caffeine_df.iterrows():
    take_time = row["taking_timestamp"]
    dose = float(row["caffeine_amount"])
    t_0 = int((take_time - start_time).total_seconds() // 3600)
    if t_0 >= len(t):
        continue

    effect = 1 / (1 + (M_c * dose / 200) * (k_a / (k_a - k_c)) *
                  (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
    effect = np.where(t < t_0, 1, effect)
    g_PD_real *= effect

P_t_real = P0_values * g_PD_real

# ===== Baseline 無咖啡因清醒度 =====
P_t_no_caffeine = P0_values.copy()

# ===== 睡覺時間設為 NaN，讓曲線斷開 =====
P_t_caffeine[~awake_flags] = np.nan
P_t_no_caffeine[~awake_flags] = np.nan
P_t_real[~awake_flags] = np.nan

# ===== 匯出計算結果至資料庫 =====
for i in range(len(time_index)):
    cursor.execute("""
        INSERT INTO alertness_data_for_visualization (user_id, timestamp, awake, "g_PD", "P0_values", "P_t_caffeine", "P_t_no_caffeine", "P_t_real")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (caffeine_df['user_id'].iloc[0], time_index[i], awake_flags[i].item(), float(g_PD[i]), float(P0_values[i]), float(P_t_caffeine[i]), float(P_t_no_caffeine[i]), float(P_t_real[i])))

# 提交變更並關閉連接
conn.commit()
cursor.close()
conn.close()

# ===== 繪製清醒度圖表 =====
x_ticks = np.arange(0, len(time_index), 3)
x_labels = [time.strftime('%m-%d %H:%M') for time in time_index[::3]]
y_max = np.nanmax([P_t_no_caffeine, P_t_caffeine, P_t_real])
y_labels = np.arange(100, y_max + 100, 50)

plt.figure(figsize=(8, 6))

# 灰色區塊標示睡眠時間
for _, row in sleep_df.iterrows():
    plt.axvspan(
        (row["start_time"] - start_time).total_seconds() / 3600,
        (row["end_time"] - start_time).total_seconds() / 3600,
        color='gray', alpha=0.2
    )

plt.plot(t, P_t_no_caffeine, label="Baseline (No Caffeine)", linestyle="--", color="gray")
plt.plot(t, P_t_caffeine, label="With Caffeine Log", color="blue")
plt.plot(t, P_t_real, label="Real-Time Intake", color="orange")  # ⬅ 新增橘色實線
plt.axhline(y=270, color='red', linestyle=':', label="Alertness Threshold = 270 ms")

plt.title("Caffeine-Adjusted Alertness Over Time (Sleep Hidden)")
plt.xlabel("Time")
plt.ylabel("Alertness P(t) (ms)")
plt.xticks(x_ticks, labels=x_labels, rotation=45)
plt.ylim(100, y_max + 100)
plt.yticks(y_labels)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
