# caffeine_recommendation.py
import numpy as np
from psycopg2.extras import execute_values
from datetime import timedelta
from typing import Dict, Optional

ALERTNESS_THRESHOLD = 270.0  # 目標上限（ms）
FORBIDDEN_HOURS_BEFORE_SLEEP = 6  # 睡前禁用時數
DOSE_STEP = 25.0              # 劑量粒度（ceil 到 25mg）
MAX_DAILY_DOSE = 300.0        # 一天上限（mg）
WINDOW_HOURS = 2              # 原本視窗長度（保留原邏輯）

# segmentation & policy
MAX_SEGMENT_HOURS = 4         # 將長 target 切成多段（每段最多 4 小時）
MIN_INTERVAL_HOURS = 4       # 同一使用者兩次建議的最小間隔（跨段也適用）
MERGE_WINDOW_SECONDS = 3600  # 合併相近建議的視窗（1 小時）

# numeric safety for exponentials
_EXP_CLIP_LOWER = -700.0
_EXP_CLIP_UPPER = 700.0

# debug
DEBUG = False


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


def _safe_phi_array(k_c, k_a, t_minus_t0):
    """
    計算 phi = exp(-k_c * dt) - exp(-k_a * dt) 的安全版本，
    使用 clipping 避免 exp overflow 並將非有限值替換為 0。
    t_minus_t0: numpy array
    """
    arg1 = np.clip(-k_c * t_minus_t0, _EXP_CLIP_LOWER, _EXP_CLIP_UPPER)
    arg2 = np.clip(-k_a * t_minus_t0, _EXP_CLIP_LOWER, _EXP_CLIP_UPPER)
    with np.errstate(over='ignore', invalid='ignore'):
        phi = np.exp(arg1) - np.exp(arg2)
    phi = np.where(np.isfinite(phi), phi, 0.0)
    return phi


def _compute_precise_dose_for_hour(
    hour_idx: int,
    P0_values: np.ndarray,
    g_PD_current: np.ndarray,
    M_c: float,
    k_a: float,
    k_c: float,
    window_hours: int = WINDOW_HOURS,
    threshold: float = ALERTNESS_THRESHOLD
) -> float:
    """
    保留原始計算方式，但無外部行為改動（ceil 等）。
    """
    best_required = 0.0
    t_len = len(P0_values)

    # guard
    if M_c is None or k_a is None or k_c is None:
        if DEBUG:
            print(f"[dose] missing params: M_c={M_c}, k_a={k_a}, k_c={k_c}")
        return 0.0
    if abs(k_a - k_c) < 1e-12:
        if DEBUG:
            print(f"[dose] k_a ≈ k_c -> skip (k_a={k_a}, k_c={k_c})")
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
        if R >= 1.0:
            continue

        delta = float(offset)
        # phi safe compute for scalar delta -> use array-like approach on scalar
        phi = np.exp(-k_c * delta) - np.exp(-k_a * delta)
        if not np.isfinite(phi) or phi <= 0:
            # if scalar phi fails numerically, try safe route (shouldn't commonly happen for small delta)
            try:
                phi = _safe_phi_array(k_c, k_a, np.array([delta]))[0]
            except Exception:
                phi = 0.0
        if phi <= 0:
            continue

        A_t = (M_c / 200.0) * (k_a / (k_a - k_c)) * phi
        if A_t <= 0:
            continue

        required = (1.0 / R - 1.0) / A_t
        if DEBUG:
            print(f"[dose] offset={offset} phi={phi}, A_t={A_t}, required={required}")

        if required > best_required:
            best_required = required

    if best_required <= 0:
        return 0.0

    steps = int(np.ceil(best_required / DOSE_STEP))
    return steps * DOSE_STEP


def _apply_dose_to_gpd(
    g_PD: np.ndarray,
    dose_mg: float,
    hour_idx: int,
    M_c: float,
    k_a: float,
    k_c: float
) -> None:
    if dose_mg <= 0:
        return
    t = np.arange(len(g_PD))
    t0 = hour_idx
    # compute phi safely for vector (t - t0)
    t_minus_t0 = t - t0
    phi = _safe_phi_array(k_c, k_a, t_minus_t0)
    factor = (M_c * dose_mg / 200.0) * (k_a / (k_a - k_c))
    den = 1.0 + factor * phi
    # guard den
    den = np.where(np.isfinite(den) & (den != 0.0), den, np.inf)
    effect = 1.0 / den
    effect = np.where(t < t0, 1.0, effect)
    g_PD *= effect


def split_into_segments(start_dt, end_dt, max_hours=MAX_SEGMENT_HOURS):
    segments = []
    cur = start_dt
    while cur < end_dt:
        seg_end = min(end_dt, cur + timedelta(hours=max_hours))
        segments.append((cur, seg_end))
        cur = seg_end
    return segments


def run_caffeine_recommendation(conn, user_params_map: Optional[Dict] = None):
    """
    user_params_map 格式: { user_id: {"M_c": float, "k_a": float, "k_c": float}, ... }
    若為 None，嘗試從 users_params 表讀取（大小寫容錯）。
    主流程：對每個 waking_period 分割成 segments，各段獨立計算，再合併排序去重，最後寫入 DB。
    """
    cur = conn.cursor()
    try:
        # 載入使用者參數（容錯處理）
        if user_params_map is None:
            user_params_map = {}
            try:
                cur.execute("SELECT user_id, m_c, k_a, k_c FROM users_params;")
                for r in cur.fetchall():
                    uid = r[0]
                    try:
                        M_c = float(r[1]) if r[1] is not None else None
                    except Exception:
                        M_c = None
                    try:
                        k_a = float(r[2]) if r[2] is not None else None
                    except Exception:
                        k_a = None
                    try:
                        k_c = float(r[3]) if r[3] is not None else None
                    except Exception:
                        k_c = None
                    user_params_map[uid] = {"M_c": M_c, "k_a": k_a, "k_c": k_c}
            except Exception:
                try:
                    cur.execute("SELECT user_id, M_c, k_a, k_c FROM users_params;")
                    for r in cur.fetchall():
                        uid = r[0]
                        try:
                            M_c = float(r[1]) if r[1] is not None else None
                        except Exception:
                            M_c = None
                        try:
                            k_a = float(r[2]) if r[2] is not None else None
                        except Exception:
                            k_a = None
                        try:
                            k_c = float(r[3]) if r[3] is not None else None
                        except Exception:
                            k_c = None
                        user_params_map[uid] = {"M_c": M_c, "k_a": k_a, "k_c": k_c}
                except Exception:
                    user_params_map = {}

        user_ids = _get_distinct_user_ids(cur)
        if not user_ids:
            if DEBUG:
                print("no users to process")
            return

        for uid in user_ids:
            latest_source_ts = _get_latest_source_ts(cur, uid)
            last_processed_ts = _get_last_processed_ts_for_rec(cur, uid)
            if latest_source_ts <= last_processed_ts:
                if DEBUG:
                    print(f"[user {uid}] no new source")
                continue

            params = user_params_map.get(uid, {"M_c": 1.1, "k_a": 1.0, "k_c": 0.5})
            M_c = params.get("M_c") if params.get("M_c") is not None else 1.1
            k_a = params.get("k_a") if params.get("k_a") is not None else 1.0
            k_c = params.get("k_c") if params.get("k_c") is not None else 0.5

            # fetch waking periods and sleep rows
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
                    print(f"[user {uid}] skipping: missing waking or sleep")
                continue

            sleep_intervals = [(r[1], r[2]) for r in sleep_rows]
            recommendations = []

            # process each waking period
            for _, target_start_time, target_end_time in waking_periods:
                # split into segments
                segments = split_into_segments(target_start_time, target_end_time, max_hours=MAX_SEGMENT_HOURS)

                # prep baseline arrays relative to target_start_time (0..24)
                total_hours = 24
                t = np.arange(0, total_hours + 1)
                P0_values = np.zeros_like(t, dtype=float)
                awake_flags = np.ones_like(t, dtype=bool)
                for h in range(24):
                    now_dt = target_start_time.replace(hour=h, minute=0, second=0, microsecond=0)
                    asleep = any(start <= now_dt < end for (start, end) in sleep_intervals)
                    awake_flags[h] = (not asleep)
                    P0_values[h] = 270.0 + _sigmoid(h) if not asleep else 270.0

                g_PD = np.ones_like(t, dtype=float)
                P_t = P0_values * g_PD

                # maintain cumulative daily dose for this waking period (to enforce MAX_DAILY_DOSE)
                cumulative_dose = 0.0
                last_recommend_abs_dt = None  # datetime of last recommendation for MIN_INTERVAL_HOURS check
                period_recommendations = []

                for seg_start, seg_end in segments:
                    # compute hour indices (relative to target_start_time)
                    start_hour = int((seg_start - target_start_time).total_seconds() // 3600)
                    end_hour = int(((seg_end - target_start_time).total_seconds() // 3600))
                    start_hour = max(0, start_hour)
                    end_hour = min(23, end_hour)

                    intake_schedule = []

                    for hour in range(start_hour, end_hour + 1):
                        recommended_time = target_start_time.replace(hour=hour, minute=0, second=0, microsecond=0)

                        # forbidden window
                        in_forbidden = any(
                            (sleep_start - timedelta(hours=FORBIDDEN_HOURS_BEFORE_SLEEP)) <= recommended_time < sleep_start
                            for (sleep_start, sleep_end) in sleep_intervals
                        )
                        if in_forbidden:
                            continue

                        if not awake_flags[hour]:
                            continue

                        if not (seg_start <= recommended_time < seg_end):
                            continue

                        if P_t[hour] <= ALERTNESS_THRESHOLD:
                            continue

                        dose_needed = _compute_precise_dose_for_hour(
                            hour_idx=hour,
                            P0_values=P0_values,
                            g_PD_current=g_PD,
                            M_c=M_c, k_a=k_a, k_c=k_c,
                            window_hours=WINDOW_HOURS,
                            threshold=ALERTNESS_THRESHOLD
                        )

                        # respect remaining daily cap (cumulative across segments)
                        remaining = MAX_DAILY_DOSE - cumulative_dose
                        dose_to_give = min(dose_needed, max(0.0, remaining))
                        if dose_to_give <= 0.0:
                            continue

                        # enforce MIN_INTERVAL_HOURS across segment boundaries
                        if last_recommend_abs_dt is not None:
                            delta_hours = (recommended_time - last_recommend_abs_dt).total_seconds() / 3600.0
                            if delta_hours < MIN_INTERVAL_HOURS:
                                if DEBUG:
                                    print(f"[user {uid}] skip {recommended_time} due to MIN_INTERVAL_HOURS (delta {delta_hours:.2f}h)")
                                continue

                        # accept recommendation
                        intake_schedule.append((uid, dose_to_give, recommended_time))
                        cumulative_dose += dose_to_give
                        last_recommend_abs_dt = recommended_time

                        # apply dose effect to g_PD and update P_t for subsequent hours
                        _apply_dose_to_gpd(g_PD, dose_to_give, hour, M_c, k_a, k_c)
                        P_t = P0_values * g_PD

                    # collect segment recommendations
                    period_recommendations.extend(intake_schedule)

                # sort & merge period_recommendations (merge near times to avoid many small splits)
                period_recommendations.sort(key=lambda x: x[2])
                merged = []
                for (u, dose, when) in period_recommendations:
                    if not merged:
                        merged.append((u, dose, when))
                        continue
                    pu, pd, pw = merged[-1]
                    if u == pu and abs((when - pw).total_seconds()) <= MERGE_WINDOW_SECONDS:
                        # merge doses (keep earlier timestamp)
                        merged[-1] = (pu, pd + dose, pw)
                    else:
                        merged.append((u, dose, when))

                # extend final recommendations
                recommendations.extend(merged)

            # after processing all waking periods for this user, sort and write to DB
            if recommendations:
                # final sort
                recommendations.sort(key=lambda x: x[2])

                # delete older recommendations for this user (preserve latest only logic as before)
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