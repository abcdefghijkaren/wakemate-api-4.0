# core/alertness_data.py
import numpy as np
from datetime import datetime, timedelta
from psycopg2.extras import execute_values
from typing import Optional

# 安全上限：避免因奇怪資料產生超大時間範圍
MAX_DAYS = 30

def _get_user_ids_for_alertness(cursor, only_user_id: Optional[str] = None):
    if only_user_id:
        return [only_user_id]
    cursor.execute("""
        SELECT DISTINCT user_id FROM (
            SELECT user_id FROM users_target_waking_period
            UNION
            SELECT user_id FROM users_real_sleep_data
            UNION
            SELECT user_id FROM users_real_time_intake
        ) AS u
    """)
    return [row[0] for row in cursor.fetchall()]

def _get_latest_source_ts(cursor, user_id):
    cursor.execute("""
        SELECT GREATEST(
            COALESCE((SELECT MAX(updated_at) FROM users_target_waking_period WHERE user_id = %s), to_timestamp(0)),
            COALESCE((SELECT MAX(updated_at) FROM users_real_sleep_data    WHERE user_id = %s), to_timestamp(0)),
            COALESCE((SELECT MAX(updated_at) FROM users_real_time_intake  WHERE user_id = %s), to_timestamp(0))
        )
    """, (user_id, user_id, user_id))
    (ts,) = cursor.fetchone()
    return ts

def _get_last_processed_ts_for_alert(cursor, user_id):
    cursor.execute("""
        SELECT COALESCE(MAX(source_data_latest_at), to_timestamp(0))
        FROM alertness_data_for_visualization
        WHERE user_id = %s
    """, (user_id,))
    (ts,) = cursor.fetchone()
    return ts

def _sigmoid(x: float, L: float = 100.0, x0: float = 14.0, k: float = 0.2) -> float:
    return L / (1.0 + np.exp(-k * (x - x0)))

def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

def run_alertness_data(conn, user_id: Optional[str] = None):
    """
    計算並覆蓋 alertness_data_for_visualization 的資料。
    若 user_id 給值，僅處理該使用者；否則處理所有使用者（但每個使用者只在其來源資料有更新時才計算）。
    """
    cur = conn.cursor()
    try:
        user_ids = _get_user_ids_for_alertness(cur, only_user_id=user_id)
        if not user_ids:
            print("缺少必要的輸入資料，無法計算清醒度。")
            return

        # 參數（如需個人化，可由外部傳入）
        M_c = 1.1
        k_a = 1.0
        k_c = 0.5
        P0_base = 270.0  # fixed P0 output (per your request)
        for uid in user_ids:
            latest_source_ts = _get_latest_source_ts(cur, uid)
            last_processed_ts = _get_last_processed_ts_for_alert(cur, uid)

            # 若來源沒更新，跳過
            if latest_source_ts <= last_processed_ts:
                # print(f"skip {uid}: latest_source_ts <= last_processed_ts")
                continue

            # 撈取來源資料（單一使用者）
            cur.execute("""
                SELECT sleep_start_time, sleep_end_time
                FROM users_real_sleep_data
                WHERE user_id = %s
                ORDER BY sleep_start_time
            """, (uid,))
            sleep_rows = cur.fetchall()  # list of (datetime, datetime)

            cur.execute("""
                SELECT target_start_time, target_end_time
                FROM users_target_waking_period
                WHERE user_id = %s
                ORDER BY target_start_time
            """, (uid,))
            target_rows = cur.fetchall()  # list of (datetime, datetime)

            # 若缺重要資料則跳過
            if not sleep_rows or not target_rows:
                print(f"skip {uid}: missing sleep or target rows")
                continue

            cur.execute("""
                SELECT taking_timestamp, caffeine_amount
                FROM users_real_time_intake
                WHERE user_id = %s
                ORDER BY taking_timestamp
            """, (uid,))
            caf_rows = cur.fetchall()  # may be empty

            cur.execute("""
                SELECT recommended_caffeine_amount, recommended_caffeine_intake_timing
                FROM recommendations_caffeine
                WHERE user_id = %s
                ORDER BY recommended_caffeine_intake_timing
            """, (uid,))
            rec_rows = cur.fetchall()  # may be empty

            # 計算時間範圍（以來源資料的最小 start 與最大 end 為準）
            try:
                min_start_candidates = []
                max_end_candidates = []
                # sleep rows: (start, end)
                min_start_candidates.append(min(r[0] for r in sleep_rows))
                max_end_candidates.append(max(r[1] for r in sleep_rows))
                # target rows: (start, end)
                min_start_candidates.append(min(r[0] for r in target_rows))
                max_end_candidates.append(max(r[1] for r in target_rows))
                # include intake timestamps if exist
                if caf_rows:
                    min_start_candidates.append(min(r[0] for r in caf_rows))
                    max_end_candidates.append(max(r[0] for r in caf_rows))
                # include rec_rows timestamps if exist
                if rec_rows:
                    # rec_rows: (amount, datetime)
                    min_start_candidates.append(min(r[1] for r in rec_rows))
                    max_end_candidates.append(max(r[1] for r in rec_rows))

                min_start = min(min_start_candidates)
                max_end = max(max_end_candidates)
            except Exception as e:
                print(f"[WARN] user {uid} time range calc failed: {e}")
                continue

            # 將起訖時間對齊整點，並加一小時保留上界
            min_start = min_start.replace(minute=0, second=0, microsecond=0)
            max_end = max_end.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

            total_hours = int((max_end - min_start).total_seconds() // 3600)
            # safety cap：避免異常資料造成超長範圍
            if total_hours < 0:
                print(f"[WARN] user {uid} computed negative total_hours, skip")
                continue
            if total_hours > 24 * MAX_DAYS:
                # cap end
                max_end = min_start + timedelta(days=MAX_DAYS)
                total_hours = int((max_end - min_start).total_seconds() // 3600)
                print(f"[WARN] user {uid} time window too large, capped to {MAX_DAYS} days")

            time_index = [min_start + timedelta(hours=i) for i in range(total_hours + 1)]
            t = np.arange(total_hours + 1)

            # 計算 awake_flags（for each hour in time_index）
            awake_flags = np.ones(len(time_index), dtype=bool)
            for i, now_dt in enumerate(time_index):
                asleep = any((start <= now_dt < end) for (start, end) in sleep_rows)
                awake_flags[i] = (not asleep)

            # ---------- P_t_no_caffeine（隨小時變化，含 sigmoid） ----------
            P_t_no_caffeine = np.zeros(len(time_index), dtype=float)
            for i, now_dt in enumerate(time_index):
                hour = now_dt.hour
                P_t_no_caffeine[i] = P0_base + _sigmoid(hour) if awake_flags[i] else P0_base

            # ---------- P0_values（保留固定輸出 270） ----------
            P0_values = np.full(len(time_index), P0_base, dtype=float)

            # ---------- g_PD_real（真實攝取） ----------
            g_PD_real = np.ones(len(time_index), dtype=float)
            if caf_rows:
                for take_time, dose in caf_rows:
                    dose = _safe_float(dose, 0.0)
                    # 若 take_time 不是 datetime，跳過（假設來源正確）
                    if not isinstance(take_time, datetime):
                        continue
                    t_0 = int((take_time - min_start).total_seconds() // 3600)
                    if t_0 >= len(t) or t_0 < -10000:
                        continue
                    effect = 1.0 / (1.0 + (M_c * dose / 200.0) * (k_a / (k_a - k_c)) *
                                   (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                    effect = np.where(t < t_0, 1.0, effect)
                    g_PD_real *= effect

            # ---------- g_PD_rec（建議攝取） ----------
            g_PD_rec = np.ones(len(time_index), dtype=float)
            if rec_rows:
                for rec_amount, rec_time in rec_rows:
                    amt = _safe_float(rec_amount, 0.0)
                    if not isinstance(rec_time, datetime):
                        continue
                    t_0 = int((rec_time - min_start).total_seconds() // 3600)
                    if t_0 >= len(t) or t_0 < -10000:
                        continue
                    effect = 1.0 / (1.0 + (M_c * amt / 200.0) * (k_a / (k_a - k_c)) *
                                   (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                    effect = np.where(t < t_0, 1.0, effect)
                    g_PD_rec *= effect

            # ---------- P_t 計算 ----------
            P_t_caffeine = P_t_no_caffeine * g_PD_rec
            P_t_real = P_t_no_caffeine * g_PD_real

            # 睡眠時間用 NaN 表示（之後轉成 None 寫入 DB，視覺化時會斷線）
            for arr in (P_t_caffeine, P_t_no_caffeine, P_t_real):
                arr[~awake_flags] = np.nan

            # 刪除舊 snapshot：**只刪除本次時間窗內**的資料（更溫和，不會刪除整個 user 的歷史）
            cur.execute("""
                DELETE FROM alertness_data_for_visualization
                WHERE user_id = %s
                  AND timestamp >= %s
                  AND timestamp <= %s
            """, (uid, min_start, max_end))

            # 準備插入資料（把 NaN -> None）
            insert_rows = []
            for i, now_dt in enumerate(time_index):
                def to_none(x):
                    return None if not np.isfinite(x) else float(x)
                insert_rows.append((
                    uid,
                    now_dt,
                    bool(awake_flags[i]),
                    float(g_PD_rec[i]) if np.isfinite(g_PD_rec[i]) else None,
                    float(g_PD_real[i]) if np.isfinite(g_PD_real[i]) else None,
                    float(P0_values[i]),
                    to_none(P_t_caffeine[i]),
                    to_none(P_t_no_caffeine[i]),
                    to_none(P_t_real[i]),
                    latest_source_ts
                ))

            # 批次插入
            execute_values(cur, """
                INSERT INTO alertness_data_for_visualization
                (user_id, timestamp, awake, g_PD_rec, g_PD_real, P0_values, P_t_caffeine, P_t_no_caffeine, P_t_real, source_data_latest_at)
                VALUES %s
            """, insert_rows)

            conn.commit()
            print(f"alertness_data: updated user {uid} rows={len(insert_rows)} (source_ts={latest_source_ts})")

    except Exception as e:
        conn.rollback()
        print(f"執行清醒度數據計算時發生錯誤: {e}")
        raise
    finally:
        cur.close()