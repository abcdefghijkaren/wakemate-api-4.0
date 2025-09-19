# personalize_params.py
import numpy as np
import psycopg2
from datetime import timedelta
from typing import Optional, Tuple, List
from statistics import mean

# Config / 超參數（可調）
ALPHA_TRAIT = 0.12        # trait EMA learning rate (kss<=4)
ALPHA_FALLBACK = 0.04     # fallback (kss>=5) learning rate, 較保守
MIN_PVT_7D = 14           # 至少 14 次 PVT 才嘗試更新 kc
MIN_DISTINCT_DAYS_7D = 7  # 過去 7 天至少 7 個不同日
KC_GRID = np.arange(0.09, 0.33 + 1e-9, 0.01)
KC_UPDATE_TOLERANCE = 0.01  # new kc 與 old kc 差異超過此值才更新

# Defaults if no users_params row exists
DEFAULTS = {
    "m_c": 1.0,
    "k_a": 1.25,
    "k_c": 0.20,
    "trait_alertness": 0.0,
    "p0_value": 270.0
}

# --- 數學函式（與現有程式一致） ---
def _sigmoid(hour: int, L: float = 100.0, x0: float = 14.0, k: float = 0.2) -> float:
    return L / (1.0 + np.exp(-k * (hour - x0)))

def _predict_rt_single(t_obs, sleep_intervals: List[Tuple], intakes: List[Tuple],
                       m_c: float, k_a: float, k_c: float) -> float:
    """
    對單一時間點 t_obs 預測 mean RT (不含 trait offset)。
    sleep_intervals: list of (start_dt, end_dt)
    intakes: list of (take_time, dose_mg)
    """
    # baseline
    asleep = any(start <= t_obs < end for (start, end) in sleep_intervals)
    base = 270.0 if asleep else (270.0 + _sigmoid(t_obs.hour))

    # g_PD_real from intakes (multiply effects)
    g = 1.0
    for (take_time, dose) in intakes:
        if take_time > t_obs:
            continue
        dt_h = (t_obs - take_time).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        # safety: ensure k_a != k_c
        if abs(k_a - k_c) < 1e-9:
            continue
        phi = np.exp(-k_c * dt_h) - np.exp(-k_a * dt_h)
        eff = 1.0 / (1.0 + (m_c * float(dose) / 200.0) * (k_a / (k_a - k_c)) * phi)
        g *= eff

    return base * g

# --- DB helper functions (assumes users_params table exists) ---
def fetch_user_params(cur, user_id):
    cur.execute("""
        SELECT m_c, k_a, k_c, trait_alertness, last_trait_update, last_kc_update, pvt_count_7d, pvt_avg_7d, p0_value
        FROM users_params
        WHERE user_id = %s
        ORDER BY updated_at DESC
        LIMIT 1
    """, (user_id,))
    r = cur.fetchone()
    if not r:
        return dict(DEFAULTS)
    m_c, k_a, k_c, trait_alertness, last_trait_update, last_kc_update, pvt_count_7d, pvt_avg_7d, p0_value = r
    return {
        "m_c": float(m_c) if m_c is not None else DEFAULTS["m_c"],
        "k_a": float(k_a) if k_a is not None else DEFAULTS["k_a"],
        "k_c": float(k_c) if k_c is not None else DEFAULTS["k_c"],
        "trait_alertness": float(trait_alertness) if trait_alertness is not None else DEFAULTS["trait_alertness"],
        "last_trait_update": last_trait_update,
        "last_kc_update": last_kc_update,
        "pvt_count_7d": int(pvt_count_7d) if pvt_count_7d is not None else 0,
        "pvt_avg_7d": float(pvt_avg_7d) if pvt_avg_7d is not None else None,
        "p0_value": float(p0_value) if p0_value is not None else DEFAULTS["p0_value"]
    }

def upsert_users_params(cur, user_id, params: dict):
    """
    Insert or update users_params row for user_id.
    Assumes users_params has a unique constraint on user_id.
    """
    cur.execute("""
        INSERT INTO users_params
        (user_id, m_c, k_a, k_c, trait_alertness, p0_value, last_trait_update, last_kc_update, pvt_count_7d, pvt_avg_7d, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE
          SET m_c = EXCLUDED.m_c,
              k_a = EXCLUDED.k_a,
              k_c = EXCLUDED.k_c,
              trait_alertness = EXCLUDED.trait_alertness,
              p0_value = EXCLUDED.p0_value,
              last_trait_update = EXCLUDED.last_trait_update,
              last_kc_update = EXCLUDED.last_kc_update,
              pvt_count_7d = EXCLUDED.pvt_count_7d,
              pvt_avg_7d = EXCLUDED.pvt_avg_7d,
              updated_at = NOW()
    """, (
        user_id,
        params.get("m_c"),
        params.get("k_a"),
        params.get("k_c"),
        params.get("trait_alertness"),
        params.get("p0_value"),
        params.get("last_trait_update"),
        params.get("last_kc_update"),
        params.get("pvt_count_7d"),
        params.get("pvt_avg_7d")
    ))

# --- data fetchers ---
def get_all_sleep_intervals(cur, user_id):
    cur.execute("""
        SELECT sleep_start_time, sleep_end_time
        FROM users_real_sleep_data
        WHERE user_id = %s
        ORDER BY sleep_start_time
    """, (user_id,))
    return cur.fetchall()

def get_all_intakes(cur, user_id):
    cur.execute("""
        SELECT taking_timestamp, caffeine_amount
        FROM users_real_time_intake
        WHERE user_id = %s
        ORDER BY taking_timestamp
    """, (user_id,))
    return cur.fetchall()

def get_pvts_last_7days(cur, user_id):
    cur.execute("""
        SELECT test_at, mean_rt, kss_level
        FROM users_pvt_results
        WHERE user_id = %s
          AND test_at >= NOW() - INTERVAL '7 days'
        ORDER BY test_at
    """, (user_id,))
    return cur.fetchall()

def update_pvt_7d_stats(cur, user_id):
    cur.execute("""
        SELECT COUNT(*), AVG(mean_rt)
        FROM users_pvt_results
        WHERE user_id = %s
          AND test_at >= NOW() - INTERVAL '7 days'
    """, (user_id,))
    cnt, avg = cur.fetchone()
    cnt = int(cnt or 0)
    avg = float(avg) if avg is not None else None

    cur.execute("""
        SELECT COUNT(DISTINCT date_trunc('day', test_at))
        FROM users_pvt_results
        WHERE user_id = %s
          AND test_at >= NOW() - INTERVAL '7 days'
    """, (user_id,))
    distinct_days = cur.fetchone()[0] or 0

    return cnt, avg, int(distinct_days)

# --- P0 (trait) 更新邏輯 ---
def update_p0_for_user(conn, user_id,
                       alpha=ALPHA_TRAIT,
                       alpha_fallback=ALPHA_FALLBACK,
                       min_pvt_7d=MIN_PVT_7D,
                       min_days=MIN_DISTINCT_DAYS_7D):
    """
    更新 trait_alertness 與 p0_value（以 KSS 篩選 + EMA）。
    條件：過去 7 天至少 min_pvt_7d 筆 PVT 且跨 >= min_days 日。
    流程：
      - 取過去 7 天 PVT，先選 kss<=4 的樣本 (S)
      - 若 S 為空，選 kss>=5 的 fallback (S_fb)
      - avg_mean_rt = 平均 mean_rt（僅 S 中資料）
      - observed_trait = avg_mean_rt - mean_group_baseline_at_these_times
      - trait_new = (1-alpha)*trait_old + alpha*observed_trait   (alpha 視 S / fallback 而定)
      - 存回 users_params (trait_alertness, p0_value = 270 + trait_new, last_trait_update)
    """
    cur = conn.cursor()
    try:
        params = fetch_user_params(cur, user_id)
        cnt_7d, avg_7d, distinct_days = update_pvt_7d_stats(cur, user_id)
        if cnt_7d < min_pvt_7d or distinct_days < min_days:
            # 更新 pvt stats only
            params.update({"pvt_count_7d": cnt_7d, "pvt_avg_7d": avg_7d})
            upsert_users_params(cur, user_id, params)
            conn.commit()
            cur.close()
            return False

        rows = get_pvts_last_7days(cur, user_id)
        if not rows:
            cur.close()
            return False

        sleep_intervals = get_all_sleep_intervals(cur, user_id)

        # filter out PVTs that occurred during sleep intervals (explicitly skip)
        filtered_rows = []
        for (test_at, mean_rt, kss) in rows:
            # skip if this test_at lies in any sleep interval
            in_sleep = any(start <= test_at < end for (start, end) in sleep_intervals)
            if in_sleep:
                continue
            filtered_rows.append((test_at, float(mean_rt), kss))

        if not filtered_rows:
            # no usable awake PVTs
            params.update({"pvt_count_7d": cnt_7d, "pvt_avg_7d": avg_7d})
            upsert_users_params(cur, user_id, params)
            conn.commit()
            cur.close()
            return False

        # choose S (kss<=4)
        S = [r for r in filtered_rows if (r[2] is not None and r[2] <= 4)]
        used_alpha = alpha
        used_label = "kss<=4"
        if not S:
            # fallback: use kss>=5
            S = [r for r in filtered_rows if (r[2] is not None and r[2] >= 5)]
            used_alpha = alpha_fallback
            used_label = "fallback_kss>=5"
            # if still empty, we can optionally decide to not update
            if not S:
                params.update({"pvt_count_7d": cnt_7d, "pvt_avg_7d": avg_7d})
                upsert_users_params(cur, user_id, params)
                conn.commit()
                cur.close()
                return False

        # compute avg_mean_rt
        avg_mean_rt = mean([r[1] for r in S])

        # compute mean_group_baseline across those times
        group_baselines = [270.0 + _sigmoid(r[0].hour) for r in S]
        mean_group_baseline = mean(group_baselines)

        # observed trait (user minus group)
        observed_trait = float(avg_mean_rt) - float(mean_group_baseline)

        trait_old = params.get("trait_alertness", 0.0)
        trait_new = (1.0 - used_alpha) * float(trait_old) + used_alpha * observed_trait

        # clip trait to reasonable bounds
        trait_new = max(min(trait_new, 120.0), -120.0)

        # compute p0_value as absolute baseline for "group reference 270"
        p0_value = 270.0 + trait_new

        params.update({
            "trait_alertness": float(trait_new),
            "p0_value": float(p0_value),
            "last_trait_update": rows[-1][0],  # use last test time used (approx)
            "pvt_count_7d": cnt_7d,
            "pvt_avg_7d": avg_7d
        })
        upsert_users_params(cur, user_id, params)
        conn.commit()
        cur.close()
        return True

    except Exception as e:
        conn.rollback()
        print(f"[update_p0_for_user] user {user_id} 發生錯誤: {e}")
        raise
    finally:
        if not cur.closed:
            cur.close()

# --- kc 更新（grid-search 最小化 MSE） ---
def maybe_update_kc_for_user(conn, user_id,
                             min_pvt_7d=MIN_PVT_7D, min_days=MIN_DISTINCT_DAYS_7D,
                             kc_grid=KC_GRID, tol=KC_UPDATE_TOLERANCE):
    """
    在已更新 trait 之後，若過去 7 天滿足 pvt 門檻，使用 grid-search 選擇使 MSE 最小的 kc。
    以目前 trait（已更新）為固定值計算預測 y_hat + trait。
    """
    cur = conn.cursor()
    try:
        # fetch params (trait 已更新可在 DB 中取得)
        params = fetch_user_params(cur, user_id)
        cnt_7d, avg_7d, distinct_days = update_pvt_7d_stats(cur, user_id)
        if cnt_7d < min_pvt_7d or distinct_days < min_days:
            # update pvt stats only
            params.update({"pvt_count_7d": cnt_7d, "pvt_avg_7d": avg_7d})
            upsert_users_params(cur, user_id, params)
            conn.commit()
            cur.close()
            return False

        # pull last 7 days rows (awake ones)
        cur.execute("""
            SELECT test_at, mean_rt
            FROM users_pvt_results
            WHERE user_id = %s
              AND test_at >= NOW() - INTERVAL '7 days'
            ORDER BY test_at
        """, (user_id,))
        rows = cur.fetchall()
        if not rows:
            cur.close()
            return False

        sleep_intervals = get_all_sleep_intervals(cur, user_id)
        intakes = get_all_intakes(cur, user_id)

        # filter rows by awake
        usable_rows = []
        for (test_at, mean_rt) in rows:
            in_sleep = any(start <= test_at < end for (start, end) in sleep_intervals)
            if in_sleep:
                continue
            usable_rows.append((test_at, float(mean_rt)))

        if not usable_rows:
            params.update({"pvt_count_7d": cnt_7d, "pvt_avg_7d": avg_7d})
            upsert_users_params(cur, user_id, params)
            conn.commit()
            cur.close()
            return False

        trait = params.get("trait_alertness", 0.0)
        m_c = params.get("m_c", DEFAULTS["m_c"])
        k_a = params.get("k_a", DEFAULTS["k_a"])
        k_c_current = params.get("k_c", DEFAULTS["k_c"])

        best_kc = k_c_current
        best_mse = None

        for kc_cand in kc_grid:
            # skip near k_a to avoid numerical instability
            if abs(k_a - kc_cand) < 1e-6:
                continue
            se = 0.0
            n = 0
            for (test_at, mean_rt) in usable_rows:
                y_hat = _predict_rt_single(test_at, sleep_intervals, intakes, m_c, k_a, kc_cand)
                y_hat_adjusted = y_hat + (trait or 0.0)
                err = (float(mean_rt) - y_hat_adjusted) ** 2
                se += err
                n += 1
            if n == 0:
                continue
            mse = se / n
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_kc = float(kc_cand)

        # decide whether to update
        if abs(best_kc - k_c_current) >= tol:
            # update directly
            cur.execute("""
                UPDATE users_params
                SET k_c = %s, last_kc_update = NOW(), pvt_count_7d = %s, pvt_avg_7d = %s, updated_at = NOW()
                WHERE user_id = %s
            """, (best_kc, cnt_7d, avg_7d, user_id))
            conn.commit()
            cur.close()
            return True
        else:
            # update pvt stats only
            params.update({"pvt_count_7d": cnt_7d, "pvt_avg_7d": avg_7d})
            upsert_users_params(cur, user_id, params)
            conn.commit()
            cur.close()
            return False

    except Exception as e:
        conn.rollback()
        print(f"[maybe_update_kc_for_user] user {user_id} 發生錯誤: {e}")
        raise
    finally:
        if not cur.closed:
            cur.close()

def update_user_params(conn, user_id):
    """
    更新指定 user_id 的個人化參數。
    內含條件檢查：7 天內至少 7 天資料、至少 14 筆 PVT。
    執行兩步驟：
      1. update_p0_for_user → 更新 trait_alertness 與 P0_value
      2. maybe_update_kc_for_user → 嘗試 grid-search 更新 kc
    回傳更新狀態，方便 API 回傳給 client。
    """
    result = {"trait_updated": False, "kc_updated": False}

    try:
        trait_updated = update_p0_for_user(conn, user_id)
        kc_updated = maybe_update_kc_for_user(conn, user_id)

        result["trait_updated"] = bool(trait_updated)
        result["kc_updated"] = bool(kc_updated)

        return result
    except Exception as e:
        print(f"[update_user_params] user {user_id} 發生錯誤: {e}")
        raise


# --- 批次入口 ---
def get_user_ids_with_pvt(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT user_id FROM users_pvt_results
    """)
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    return ids

def update_all_users(conn, user_ids: Optional[List] = None):
    """
    更新所有 user 的 P0/trait 與在條件滿足時更新 kc。
    呼叫順序：先更新 trait (P0)，再嘗試更新 kc（以 trait 固定）。
    """
    cur = conn.cursor()
    try:
        if user_ids is None:
            user_ids = get_user_ids_with_pvt(conn)
        for uid in user_ids:
            print(f"[personalize] Updating P0/trait for user {uid} ...")
            update_p0_for_user(conn, uid)
            print(f"[personalize] Maybe updating kc for user {uid} ...")
            updated = maybe_update_kc_for_user(conn, uid)
            print(f"[personalize] kc updated: {updated}")
    finally:
        cur.close()