# personalize_params.py
import numpy as np
import psycopg2
from datetime import timedelta
from typing import Optional, Tuple, List

# Config / 超參數（可調）
ALPHA_TRAIT = 0.12        # trait EMA learning rate (每次 PVT 更新)
MIN_PVT_7D = 14           # 至少 14 次 PVT 才嘗試更新 kc
MIN_DISTINCT_DAYS_7D = 7  # 過去 7 天至少 7 個不同日
KC_GRID = np.arange(0.09, 0.33 + 1e-9, 0.01)
KC_UPDATE_TOLERANCE = 0.01  # new kc 與 old kc 差異超過此值才更新
DEFAULTS = {
    "m_c": 1.1,
    "k_a": 1.2,
    "k_c": 0.20,
    "trait_alertness": 0.0
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
            # fallback: skip this intake (should rarely happen)
            continue
        phi = np.exp(-k_c * dt_h) - np.exp(-k_a * dt_h)
        eff = 1.0 / (1.0 + (m_c * float(dose) / 200.0) * (k_a / (k_a - k_c)) * phi)
        g *= eff

    return base * g

# --- DB helper functions ---
def fetch_user_params(cur, user_id):
    cur.execute("""
        SELECT m_c, k_a, k_c, trait_alertness, last_trait_update, last_kc_update, pvt_count_7d, pvt_avg_7d
        FROM users_params
        WHERE user_id = %s
        ORDER BY updated_at DESC
        LIMIT 1
    """, (user_id,))
    r = cur.fetchone()
    if not r:
        return dict(DEFAULTS)
    m_c, k_a, k_c, trait_alertness, last_trait_update, last_kc_update, pvt_count_7d, pvt_avg_7d = r
    return {
        "m_c": float(m_c) if m_c is not None else DEFAULTS["m_c"],
        "k_a": float(k_a) if k_a is not None else DEFAULTS["k_a"],
        "k_c": float(k_c) if k_c is not None else DEFAULTS["k_c"],
        "trait_alertness": float(trait_alertness) if trait_alertness is not None else DEFAULTS["trait_alertness"],
        "last_trait_update": last_trait_update,
        "last_kc_update": last_kc_update,
        "pvt_count_7d": int(pvt_count_7d) if pvt_count_7d is not None else 0,
        "pvt_avg_7d": float(pvt_avg_7d) if pvt_avg_7d is not None else None
    }

def upsert_users_params(cur, user_id, params: dict):
    """
    Insert or update users_params row for user_id.
    Assumes users_params has a unique constraint on user_id.
    """
    cur.execute("""
        INSERT INTO users_params
        (user_id, m_c, k_a, k_c, trait_alertness, last_trait_update, last_kc_update, pvt_count_7d, pvt_avg_7d, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE
          SET m_c = EXCLUDED.m_c,
              k_a = EXCLUDED.k_a,
              k_c = EXCLUDED.k_c,
              trait_alertness = EXCLUDED.trait_alertness,
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
        params.get("last_trait_update"),
        params.get("last_kc_update"),
        params.get("pvt_count_7d"),
        params.get("pvt_avg_7d")
    ))


# --- 主邏輯函式 ---
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


def get_pvts_since(cur, user_id, since_ts):
    if since_ts is None:
        cur.execute("""
            SELECT test_at, mean_rt
            FROM users_pvt_results
            WHERE user_id = %s
            ORDER BY test_at
        """, (user_id,))
    else:
        cur.execute("""
            SELECT test_at, mean_rt
            FROM users_pvt_results
            WHERE user_id = %s
              AND test_at > %s
            ORDER BY test_at
        """, (user_id, since_ts))
    return cur.fetchall()


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

def update_trait_for_user(conn, user_id, alpha=ALPHA_TRAIT):
    cur = conn.cursor()
    try:
        params = fetch_user_params(cur, user_id)
        sleep_intervals = get_all_sleep_intervals(cur, user_id)
        intakes = get_all_intakes(cur, user_id)

        last_trait_update = params.get("last_trait_update")
        pvt_rows = get_pvts_since(cur, user_id, last_trait_update)

        trait = params.get("trait_alertness", 0.0)

        if not pvt_rows:
            # still update pvt_7d stats
            cnt, avg, distinct_days = update_pvt_7d_stats(cur, user_id)
            params.update({"pvt_count_7d": cnt, "pvt_avg_7d": avg})
            upsert_users_params(cur, user_id, params)
            conn.commit()
            cur.close()
            return

        # 逐筆 PVT 做 EMA 更新
        last_used_ts = last_trait_update
        for (test_at, mean_rt) in pvt_rows:
            # predicted RT at test_at using current m_c,k_a,k_c (no trait)
            pred_no_trait = _predict_rt_single(test_at, sleep_intervals, intakes,
                                               params["m_c"], params["k_a"], params["k_c"])
            # update trait: aim pred + trait ~= obs  => trait_new = (1-alpha)*trait_old + alpha*(obs - pred_no_trait)
            trait = (1.0 - alpha) * trait + alpha * (float(mean_rt) - pred_no_trait)
            last_used_ts = test_at

        # 更新 pvt_7d stats too
        cnt, avg, distinct_days = update_pvt_7d_stats(cur, user_id)
        params.update({
            "trait_alertness": float(trait),
            "last_trait_update": last_used_ts,
            "pvt_count_7d": cnt,
            "pvt_avg_7d": avg
        })
        upsert_users_params(cur, user_id, params)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[update_trait_for_user] user {user_id} 發生錯誤: {e}")
        raise
    finally:
        cur.close()


def maybe_update_kc_for_user(conn, user_id, min_pvt_7d=MIN_PVT_7D, min_days=MIN_DISTINCT_DAYS_7D,
                            kc_grid=KC_GRID, tol=KC_UPDATE_TOLERANCE):
    """
    只有在過去 7 天有 >= min_pvt_7d 筆 PVT 且跨至少 min_days 天時才嘗試更新 kc。
    使用 grid-search 最小化 MSE（固定 trait）。
    """
    cur = conn.cursor()
    try:
        # load params (trait 已在 earlier step 更新)
        params = fetch_user_params(cur, user_id)
        cnt_7d, avg_7d, distinct_days = update_pvt_7d_stats(cur, user_id)
        if cnt_7d < min_pvt_7d or distinct_days < min_days:
            # 更新 pvt stats in DB anyway
            params.update({"pvt_count_7d": cnt_7d, "pvt_avg_7d": avg_7d})
            upsert_users_params(cur, user_id, params)
            conn.commit()
            cur.close()
            return False

        # 取得過去 7 天的 PVT rows
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

        trait = params.get("trait_alertness", 0.0)
        m_c = params.get("m_c", DEFAULTS["m_c"])
        k_a = params.get("k_a", DEFAULTS["k_a"])
        k_c_current = params.get("k_c", DEFAULTS["k_c"])

        best_kc = k_c_current
        best_mse = None

        for kc_cand in kc_grid:
            # skip if kc near k_a (numerical instability)
            if abs(k_a - kc_cand) < 1e-6:
                continue
            se = 0.0
            n = 0
            for (test_at, mean_rt) in rows:
                y_hat = _predict_rt_single(test_at, sleep_intervals, intakes, m_c, k_a, kc_cand)
                # include current trait as offset
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
            params.update({"k_c": float(best_kc), "last_kc_update": psycopg2.sql.Literal('NOW()')})
            # We will execute upsert with proper last_kc_update = now()
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
    更新所有 user 的 trait（每次 PVT 後）並在條件滿足時嘗試更新 kc。
    """
    cur = conn.cursor()
    try:
        if user_ids is None:
            user_ids = get_user_ids_with_pvt(conn)
        for uid in user_ids:
            print(f"Updating trait for user {uid} ...")
            update_trait_for_user(conn, uid, alpha=ALPHA_TRAIT)
            print(f"Maybe updating kc for user {uid} ...")
            updated = maybe_update_kc_for_user(conn, uid)
            print(f"kc updated: {updated}")
    finally:
        cur.close()


# Example usage:
if __name__ == "__main__":
    # 設定你的 DB 連線
    conn = psycopg2.connect("dbname=yourdb user=youruser password=yourpw host=yourhost")
    try:
        update_all_users(conn)
    finally:
        conn.close()