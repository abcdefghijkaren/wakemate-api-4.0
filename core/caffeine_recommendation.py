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

# Merge / dedupe config (seconds)
MERGE_SECONDS = 3600  # 合併 1 小時內的推薦成一筆


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
    if abs(k_a - k_c) < 1e-12:
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
        # compute phi safely
        try:
            phi = np.exp(-k_c * delta) - np.exp(-k_a * delta)
        except OverflowError:
            # numerical protection: if overflow, skip this offset
            if DEBUG:
                print(f"[dose] phi overflow at delta={delta}, k_c={k_c}, k_a={k_a}")
            continue

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
    MIN_EFFECTIVE_FRACTION = 0.1
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
    # do nothing for zero dose
    if dose_mg <= 0:
        return
    # guard against extremely out-of-range hour_idx to avoid exp overflow
    # hour_idx is relative to the g_PD array index
    t = np.arange(len(g_PD))
    t0 = hour_idx
    # If t0 is far in the future relative to t, the exponent can overflow.
    # Only apply when t0 is within a reasonable range.
    if t0 > len(t) + 1000 or t0 < -1000:
        return
    # compute effect with vectorized ops (numpy handles large magnitudes)
    # but we protect against invalid operations by using np.errstate
    with np.errstate(over='ignore', invalid='ignore'):
        effect = 1.0 / (1.0 + (m_c * dose_mg / 200.0) * (k_a / (k_a - k_c)) *
                        (np.exp(-k_c * (t - t0)) - np.exp(-k_a * (t - t0))))
    # replace invalid / nan with 1.0 (no effect) to be conservative
    effect = np.where(np.isfinite(effect), effect, 1.0)
    effect = np.where(t < t0, 1.0, effect)
    g_PD *= effect


def _merge_recommendations_list(recs):
    """
    recs: list of (user_id, dose, when)
    merge entries with same user and within MERGE_SECONDS into a single record by summing dose.
    Keep the earlier timestamp as the representative time.
    """
    if not recs:
        return []
    recs_sorted = sorted(recs, key=lambda x: (x[0], x[2]))
    merged = []
    for uid_, dose_, when_ in recs_sorted:
        if not merged:
            merged.append((uid_, dose_, when_))
            continue
        puid, pdose, pwhen = merged[-1]
        if uid_ == puid and abs((when_ - pwhen).total_seconds()) <= MERGE_SECONDS:
            # merge into previous; keep earlier timestamp (pwhen)
            merged[-1] = (puid, pdose + dose_, pwhen)
        else:
            merged.append((uid_, dose_, when_))
    return merged


def run_caffeine_recommendation(conn, user_params_map: Optional[Dict] = None):
    """
    user_params_map 格式:
      { user_id: {"m_c": float, "k_a": float, "k_c": float, "trait": float, "p0_value": float}, ... }

    DB 寫入策略：
    - 不刪歷史推薦
    - 每次對某個 user 產生新推薦時：
        1) 將該 user 目前所有「未來」且 is_active=true 的推薦標為 inactive
        2) 插入本次 run 的推薦（同一個 run_id，is_active=true）
    """
    from uuid import uuid4

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
                        "m_c": float(r[1]) if r[1] is not None else 1.0,
                        "k_a": float(r[2]) if r[2] is not None else 1.25,
                        "k_c": float(r[3]) if r[3] is not None else 0.20,
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

            params = user_params_map.get(
                uid,
                {"m_c": 1.0, "k_a": 1.25, "k_c": 0.20, "trait": 0.0, "p0_value": 270.0}
            )
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

            # fetch actual intakes for this user once (used to initialize g_PD and to skip exact-time recommendations)
            cur.execute("""
                SELECT taking_timestamp, caffeine_amount
                FROM users_real_time_intake
                WHERE user_id = %s
                ORDER BY taking_timestamp
            """, (uid,))
            caf_rows = cur.fetchall()  # list of (taking_timestamp, caffeine_amount)

            if not waking_periods or not sleep_rows:
                continue

            sleep_intervals = [(r[1], r[2]) for r in sleep_rows]

            # collect recommendations across all waking periods
            all_insert_rows = []

            for _, target_start_time, target_end_time in waking_periods:
                # For each waking period, simulate 24h grid anchored at target_start_time
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

                # initialize g_PD and apply existing real intakes so the simulation reflects reality
                g_PD = np.ones_like(t, dtype=float)
                intake_time_to_amount = {}

                for (take_ts, amt) in caf_rows:
                    if take_ts is None:
                        continue
                    try:
                        hour_idx = int((take_ts - target_start_time).total_seconds() // 3600)
                    except Exception:
                        continue

                    # apply only when hour_idx is within reasonable window to avoid numerical instability
                    if -len(t) <= hour_idx < len(t) + 1:
                        try:
                            _apply_dose_to_gpd(g_PD, float(amt), hour_idx, m_c, k_a, k_c)
                        except Exception:
                            pass

                    # record mapping for exact-time skip logic (sum if multiple same ts)
                    if take_ts in intake_time_to_amount:
                        intake_time_to_amount[take_ts] += float(amt)
                    else:
                        intake_time_to_amount[take_ts] = float(amt)

                P_t = P0_values * g_PD

                daily_dose = 0.0
                intake_schedule = []

                # iterate hours and propose doses
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

                    # 如果該 exact timestamp 已有使用者實際攝取，則跳過推薦（避免重複）
                    if recommended_time in intake_time_to_amount:
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

                    # apply proposed dose to g_PD for subsequent hours in this window's simulation
                    _apply_dose_to_gpd(g_PD, dose_to_give, hour, m_c, k_a, k_c)
                    P_t = P0_values * g_PD

                # filter intake_schedule to the target window (should already be)
                period_recommendations = [
                    (uid, dose, when)
                    for (uid, dose, when) in intake_schedule
                    if target_start_time <= when <= target_end_time
                ]

                # merge close/duplicate recs within this period
                merged = _merge_recommendations_list(period_recommendations)

                # prepare insert rows for this period
                for (u, d, when) in merged:
                    all_insert_rows.append((u, d, when, latest_source_ts))

            # after processing all waking_periods for this user, do a final dedupe/merge across periods
            all_insert_rows_merged = _merge_recommendations_list([(r[0], r[1], r[2]) for r in all_insert_rows])

            if all_insert_rows_merged:
                # 只存未來
                cur.execute("SELECT NOW()")
                db_now = cur.fetchone()[0]

                values = [(r[0], r[1], r[2], latest_source_ts) for r in all_insert_rows_merged]
                values = [v for v in values if v[2] is not None and v[2] >= db_now]

                if values:
                    # 這次「整批推薦」共用同一個 run_id
                    this_run_id = uuid4()

                    # 先把該 user 目前未來的 active 推薦全部關掉（保留歷史，不刪）
                    cur.execute("""
                        UPDATE recommendations_caffeine
                        SET is_active = FALSE
                        WHERE user_id = %s
                          AND is_active = TRUE
                          AND recommended_caffeine_intake_timing >= NOW()
                    """, (uid,))

                    # 插入本次 run 的推薦：is_active=true + run_id
                    # 使用 ON CONFLICT DO NOTHING（最安全，不依賴你現在 unique constraint 長怎樣）
                    execute_values(
                        cur,
                        """
                        INSERT INTO recommendations_caffeine
                        (user_id, recommended_caffeine_amount, recommended_caffeine_intake_timing,
                         source_data_latest_at, run_id, is_active)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                        """,
                        [(u, d, when, src_ts, this_run_id, True) for (u, d, when, src_ts) in values]
                    )

                    conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"執行咖啡因建議計算時發生錯誤: {e}")
    finally:
        cur.close()