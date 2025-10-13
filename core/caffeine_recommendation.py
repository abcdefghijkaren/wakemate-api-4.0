# caffeine_recommendation.py
import numpy as np
from psycopg2.extras import execute_values
from datetime import timedelta
from typing import Dict, Optional

ALERTNESS_THRESHOLD = 270.0  # 目標上限（ms）
FORBIDDEN_HOURS_BEFORE_SLEEP = 6
DOSE_STEP = 25.0
MAX_DAILY_DOSE = 300.0
WINDOW_HOURS = 2


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
    # debug 開關：設 True 會 print 詳細內部變數（部署時設 False）
    DEBUG = False

    best_required = 0.0
    t_len = len(P0_values)

    # sanity checks
    if k_a == k_c:
        # 避免除以零
        if DEBUG:
            print(f"[dose] k_a == k_c ({k_a}) — 返回 0")
        return 0.0
    if m_c is None or m_c == 0:
        if DEBUG:
            print(f"[dose] m_c is zero/None ({m_c}) — 返回 0")
        return 0.0

    for offset in range(1, window_hours + 1):
        t_j = hour_idx + offset
        if t_j >= t_len:
            break

        base = P0_values[t_j] * g_PD_current[t_j]
        # debug
        if DEBUG:
            print(f"[dose] hour={hour_idx} offset={offset} t_j={t_j} base={base}")

        if base <= 0:
            continue

        R = threshold / base
        # debug
        if DEBUG:
            print(f"[dose] R={R}")

        # only proceed if R < 1 (i.e. base > threshold)
        if R >= 1.0:
            continue

        delta = float(offset)
        phi = np.exp(-k_c * delta) - np.exp(-k_a * delta)
        if DEBUG:
            print(f"[dose] phi={phi}")

        if phi <= 0:
            continue

        A_t = (m_c / 200.0) * (k_a / (k_a - k_c)) * phi
        # guard A_t tiny or negative
        if A_t <= 0:
            if DEBUG:
                print(f"[dose] A_t non-positive: {A_t}")
            continue

        required = (1.0 / R - 1.0) / A_t
        if DEBUG:
            print(f"[dose] required={required}")

        if required > best_required:
            best_required = required

    # If nothing needed
    if best_required <= 0:
        return 0.0

    # Protect against tiny fractional required values that would be rounded up to one DOSE_STEP
    # (avoid returning 25 mg for trivial tiny requirements)
    MIN_EFFECTIVE_FRACTION = 0.5  # if required < DOSE_STEP * 0.5 -> treat as 0
    if best_required < DOSE_STEP * MIN_EFFECTIVE_FRACTION:
        if DEBUG:
            print(f"[dose] best_required {best_required} < {DOSE_STEP * MIN_EFFECTIVE_FRACTION} -> treat as 0")
        return 0.0

    steps = int(np.ceil(best_required / DOSE_STEP))
    return steps * DOSE_STEP



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
            print("沒有可處理的使用者（沒有清醒/睡眠資料）")
            return

        for uid in user_ids:
            latest_source_ts = _get_latest_source_ts(cur, uid)
            last_processed_ts = _get_last_processed_ts_for_rec(cur, uid)
            if latest_source_ts <= last_processed_ts:
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

                    intake_schedule.append((uid, dose_to_give, recommended_time))
                    daily_dose += dose_to_give

                    _apply_dose_to_gpd(g_PD, dose_to_give, hour, m_c, k_a, k_c)
                    P_t = P0_values * g_PD

                filtered = [
                    (uid, dose, when)
                    for (uid, dose, when) in intake_schedule
                    if target_start_time <= when <= target_end_time
                ]
                recommendations.extend(filtered)

            if recommendations:
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

    except Exception as e:
        conn.rollback()
        print(f"執行咖啡因建議計算時發生錯誤: {e}")
    finally:
        cur.close()