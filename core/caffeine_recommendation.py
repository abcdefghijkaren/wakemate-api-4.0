# caffeine_recommendation.py (修補版)
import numpy as np
from psycopg2.extras import execute_values
from datetime import timedelta
from typing import Dict, Optional

# ---------- Config / 超參數（可調） ----------
ALERTNESS_THRESHOLD = 270.0  # 目標上限（ms）
FORBIDDEN_HOURS_BEFORE_SLEEP = 6
DOSE_STEP = 25.0
MAX_DAILY_DOSE = 300.0

# 改為看更長的未來窗，避免把長時間清醒需求切成很多小劑量
WINDOW_HOURS = 4

# 同一使用者兩次建議的最小間隔（小於此值會跳過較近期的小劑量）
MIN_INTERVAL_HOURS = 3

# debug 開關（設 True 可印出內部變數，協助除錯）
DEBUG = False
# -----------------------------------------------


def _get_distinct_user_ids(cursor):
    cursor.execute("""
        SELECT DISTINCT user_id FROM (
            SELECT user_id FROM users_target_waking_period
            UNION
            SELECT user_id FROM users_real_sleep_data
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


def _get_last_processed_ts_for_rec(cursor, user_id):
    cursor.execute("""
        SELECT COALESCE(MAX(source_data_latest_at), to_timestamp(0))
        FROM recommendations_caffeine
        WHERE user_id = %s
    """, (user_id,))
    (ts,) = cursor.fetchone()
    return ts


def _sigmoid(x: float, L: float = 100.0, x0: float = 14.0, k: float = 0.2) -> float:
    return L / (1 + np.exp(-k * (x - x0)))


def _is_in_forbidden_window(ts, sleep_intervals) -> bool:
    for (sleep_start, _sleep_end) in sleep_intervals:
        if (sleep_start - timedelta(hours=FORBIDDEN_HOURS_BEFORE_SLEEP)) <= ts < sleep_start:
            return True
    return False


def _compute_precise_dose_for_hour(
    hour_idx: int,
    P0_values: np.ndarray,
    g_PD_current: np.ndarray,
    m_c: float,
    k_a: float,
    k_c: float,
    window_hours: int = WINDOW_HOURS,
    threshold: float = ALERTNESS_THRESHOLD
) -> float:
    """
    計算在 hour_idx 時刻需要的劑量（mg）。
    修改重點：
      - 增加窗長預設為 WINDOW_HOURS（檔頭調整）
      - 更保守的四捨五入：只有當 required >= DOSE_STEP 時才建議最小一步 (25 mg)
      - 使用 round 而非 ceil，讓步數較接近直覺
      - DEBUG 可印內部變數
    """
    best_required = 0.0
    t_len = len(P0_values)

    # sanity guards
    if abs(k_a - k_c) < 1e-9:
        if DEBUG:
            print(f"[dose] skip: k_a ≈ k_c ({k_a})")
        return 0.0
    if m_c is None or m_c == 0:
        if DEBUG:
            print(f"[dose] skip: m_c invalid ({m_c})")
        return 0.0

    for offset in range(1, window_hours + 1):
        t_j = hour_idx + offset
        if t_j >= t_len:
            break

        base = P0_values[t_j] * g_PD_current[t_j]
        if DEBUG:
            print(f"[dose] hour={hour_idx} offset={offset} t_j={t_j} base={base}")

        if base <= 0:
            continue

        R = threshold / base
        if DEBUG:
            print(f"[dose] R={R}")

        # 只有當 base > threshold（即 R < 1）才需要考慮補強
        if R >= 1.0:
            continue

        delta = float(offset)
        phi = np.exp(-k_c * delta) - np.exp(-k_a * delta)
        if DEBUG:
            print(f"[dose] phi={phi}")

        if phi <= 0:
            continue

        A_t = (m_c / 200.0) * (k_a / (k_a - k_c)) * phi
        if DEBUG:
            print(f"[dose] A_t={A_t}")

        if A_t <= 0:
            continue

        required = (1.0 / R - 1.0) / A_t
        if DEBUG:
            print(f"[dose] required={required}")

        if required > best_required:
            best_required = required

    if best_required <= 0:
        return 0.0

    # 更保守的策略：若 required < DOSE_STEP，視為不需要（避免把 1-24 mg 抬到 25）
    if best_required < DOSE_STEP:
        if DEBUG:
            print(f"[dose] best_required {best_required:.3f} < DOSE_STEP ({DOSE_STEP}) -> 0")
        return 0.0

    # 回傳最接近的步數（round）。若希望偏向保守可改成 ceil。
    steps = int(round(best_required / DOSE_STEP))
    if steps <= 0:
        steps = 1
    dose = steps * DOSE_STEP
    if DEBUG:
        print(f"[dose] best_required={best_required:.3f} -> steps={steps}, dose={dose}")
    return dose


def _apply_dose_to_gpd(
    g_PD: np.ndarray,
    dose_mg: float,
    hour_idx: int,
    m_c: float,
    k_a: float,
    k_c: float
) -> None:
    if dose_mg <= 0:
        return
    t = np.arange(len(g_PD))
    t0 = hour_idx
    effect = 1.0 / (1.0 + (m_c * dose_mg / 200.0) * (k_a / (k_a - k_c)) *
                    (np.exp(-k_c * (t - t0)) - np.exp(-k_a * (t - t0))))
    effect = np.where(t < t0, 1.0, effect)
    g_PD *= effect


def run_caffeine_recommendation(conn, user_params_map: Optional[Dict] = None):
    """
    user_params_map 格式: 
      { user_id: {"m_c": float, "k_a": float, "k_c": float, "trait": float, "p0_value": float}, ... }
    行為改動：
      - 使用 WINDOW_HOURS（預設 4）來計算 required
      - enforce MIN_INTERVAL_HOURS（預設 3）避免短時間重複給小劑量
      - 若 recommendations 非空則刪除舊推薦（基於 source_data_latest_at 邏輯保留）
    """
    cur = conn.cursor()
    try:
        # 載入使用者參數
        if user_params_map is None:
            user_params_map = {}
            try:
                cur.execute("""
                    SELECT user_id, m_c, k_a, k_c, trait_alertness, p0_value 
                    FROM users_params;
                """)
                for r in cur.fetchall():
                    user_params_map[r[0]] = {
                        "m_c": float(r[1]),
                        "k_a": float(r[2]),
                        "k_c": float(r[3]),
                        "trait": float(r[4] or 0.0),
                        "p0_value": float(r[5] or 270.0)
                    }
            except Exception:
                user_params_map = {}

        user_ids = _get_distinct_user_ids(cur)
        if not user_ids:
            if DEBUG:
                print("沒有可處理的使用者（沒有清醒/睡眠資料）")
            return

        for uid in user_ids:
            latest_source_ts = _get_latest_source_ts(cur, uid)
            last_processed_ts = _get_last_processed_ts_for_rec(cur, uid)
            if latest_source_ts <= last_processed_ts:
                if DEBUG:
                    print(f"[user {uid}] no new source data (latest_source_ts <= last_processed_ts)")
                continue

            params = user_params_map.get(uid, {"m_c": 1.0, "k_a": 1.25, "k_c": 0.20, "trait": 0.0, "p0_value": 270.0})
            m_c, k_a, k_c, trait, p0_value = (
                params["m_c"], params["k_a"], params["k_c"], params["trait"], params["p0_value"]
            )

            # 取該使用者資料
            cur.execute("""
                SELECT user_id, target_start_time, target_end_time
                FROM users_target_waking_period
                WHERE user_id = %s
                ORDER BY target_start_time
            """, (uid,))
            waking_periods = cur.fetchall()

            cur.execute("""
                SELECT user_id, sleep_start_time, sleep_end_time
                FROM users_real_sleep_data
                WHERE user_id = %s
                ORDER BY sleep_start_time
            """, (uid,))
            sleep_rows = cur.fetchall()

            if not waking_periods or not sleep_rows:
                if DEBUG:
                    print(f"[user {uid}] skipping: missing waking_periods or sleep_rows")
                continue

            sleep_intervals = [(r[1], r[2]) for r in sleep_rows]
            recommendations = []

            for _, target_start_time, target_end_time in waking_periods:
                total_hours = 24
                t = np.arange(0, total_hours + 1)

                P0_values = np.zeros_like(t, dtype=float)
                awake_flags = np.ones_like(t, dtype=bool)

                for h in range(24):
                    now_dt = target_start_time.replace(hour=h, minute=0, second=0, microsecond=0)
                    asleep = any(start <= now_dt < end for (start, end) in sleep_intervals)
                    awake_flags[h] = (not asleep)

                    if asleep:
                        # 睡眠：固定 baseline + trait
                        P0_values[h] = 270.0 + trait
                    else:
                        # 清醒：baseline + circadian + trait
                        P0_values[h] = (270.0 + _sigmoid(h)) + trait

                g_PD = np.ones_like(t, dtype=float)
                P_t = P0_values * g_PD

                daily_dose = 0.0
                intake_schedule = []
                last_recommend_hour = -999  # 用整數 hour index 記錄上次建議位置（相對於 target_start_time 的 hour）

                for hour in range(24):
                    recommended_time = target_start_time.replace(hour=hour, minute=0, second=0, microsecond=0)

                    # 禁止睡前 6 小時
                    in_forbidden = any(
                        (sleep_start - timedelta(hours=6)) <= recommended_time < sleep_start
                        for (sleep_start, sleep_end) in sleep_intervals
                    )
                    if in_forbidden:
                        continue

                    if not awake_flags[hour]:
                        continue

                    if not (target_start_time <= recommended_time <= target_end_time):
                        continue

                    if P_t[hour] <= ALERTNESS_THRESHOLD:
                        continue

                    dose_needed = _compute_precise_dose_for_hour(
                        hour_idx=hour,
                        P0_values=P0_values,
                        g_PD_current=g_PD,
                        m_c=m_c, k_a=k_a, k_c=k_c,
                        window_hours=WINDOW_HOURS,
                        threshold=ALERTNESS_THRESHOLD
                    )

                    remaining = MAX_DAILY_DOSE - daily_dose
                    dose_to_give = min(dose_needed, max(0.0, remaining))
                    if dose_to_give <= 0.0:
                        continue

                    # enforce minimal spacing between recommendations
                    if last_recommend_hour != -999:
                        if (hour - last_recommend_hour) < MIN_INTERVAL_HOURS:
                            if DEBUG:
                                print(f"[user {uid}] skip hour {hour} due to MIN_INTERVAL_HOURS (last {last_recommend_hour})")
                            continue

                    # accept this recommendation
                    intake_schedule.append((uid, dose_to_give, recommended_time))
                    daily_dose += dose_to_give
                    last_recommend_hour = hour

                    _apply_dose_to_gpd(g_PD, dose_to_give, hour, m_c, k_a, k_c)
                    P_t = P0_values * g_PD

                filtered = [
                    (uid, dose, when)
                    for (uid, dose, when) in intake_schedule
                    if target_start_time <= when <= target_end_time
                ]
                recommendations.extend(filtered)

            if recommendations:
                # 覆蓋策略保留：刪除舊的（較舊 source_data_latest_at 的）並插入新的推薦
                cur.execute("""
                    DELETE FROM recommendations_caffeine
                    WHERE user_id = %s
                      AND (source_data_latest_at IS NULL OR source_data_latest_at < %s)
                """, (uid, latest_source_ts))

                execute_values(
                    cur,
                    """
                    INSERT INTO recommendations_caffeine
                    (user_id, recommended_caffeine_amount, recommended_caffeine_intake_timing, source_data_latest_at)
                    VALUES %s
                    """,
                    [(r[0], r[1], r[2], latest_source_ts) for r in recommendations]
                )
                conn.commit()
                if DEBUG:
                    print(f"[user {uid}] inserted {len(recommendations)} recommendations")

    except Exception as e:
        conn.rollback()
        print(f"執行咖啡因建議計算時發生錯誤: {e}")
        raise
    finally:
        cur.close()