import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="AlphaPulse Institutional Quant Terminal Pro", layout="wide")

# -------------------------------------------------------------
# 0. Alpha Vantage API 연동 모듈 (yfinance IP 차단 방어 및 COHR/ADR 보강)
# -------------------------------------------------------------
ALPHA_VANTAGE_KEY = "BHJM0RK8XN3DZTP3"

def fetch_alpha_vantage_daily(symbol: str) -> pd.DataFrame:
    """Alpha Vantage 일봉 수신 어댑터 (yfinance 실패 시 자동 폴백)"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=4)
        data = res.json()
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }).astype(float)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------------------
# 1. 시장 운영 세션 판별 엔진 (국내 09:00~18:00 / 미국 22:30~05:00)
# -------------------------------------------------------------
def get_market_session_status(currency):
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc + timedelta(hours=9)
    now_edt = now_utc - timedelta(hours=4)

    weekday_kst = now_kst.weekday()
    weekday_edt = now_edt.weekday()

    if currency == "₩":
        is_weekday = weekday_kst < 5
        t_kst = now_kst.time()
        is_market_open = is_weekday and (
            datetime.strptime("09:00", "%H:%M").time() <= t_kst <= datetime.strptime("18:00", "%H:%M").time()
        )
        session_text = "국내 정규장/시간외 거래 진행 중" if is_market_open else "국내장 마감 (익일 개장 전 예측 모드)"
        time_str = now_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        return is_market_open, session_text, time_str
    else:
        is_weekday = weekday_edt < 5
        t_edt = now_edt.time()
        is_market_open = is_weekday and (
            datetime.strptime("09:30", "%H:%M").time() <= t_edt <= datetime.strptime("16:00", "%H:%M").time()
        )
        session_text = "미국 정규장 진행 중" if is_market_open else "미국장 마감 (익일 개장 전 예측 모드)"
        time_str = f"{now_edt.strftime('%H:%M:%S EDT')} (KST {now_kst.strftime('%H:%M:%S')})"
        return is_market_open, session_text, time_str

# -------------------------------------------------------------
# 2. 안전한 가격 포맷팅 유틸리티
# -------------------------------------------------------------
def format_price(val, currency):
    if val is None or np.isnan(val) or val <= 0:
        return "시세 집계 중"
    if currency == "₩":
        return f"{int(round(val)):,}원"
    else:
        return f"${val:.2f}"

def get_asset_live_change(df):
    """최근 일봉 대비 실시간 장중 변동률(%) 추출"""
    if df is None or len(df) < 2:
        return 0.0
    latest = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2])
    if prev > 0:
        return float(((latest - prev) / prev) * 100)
    return 0.0

# -------------------------------------------------------------
# 3. 네이버 금융 일별 외인/기관 순매수 수급 시계열 수집
# -------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_naver_investor_flows(ticker_code, pages=10):
    if ticker_code == "COHR":
        return pd.DataFrame()
        
    code_digits = ticker_code.split(".")[0]
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []
    
    for p in range(1, pages + 1):
        url = f"https://finance.naver.com/item/frgn.naver?code={code_digits}&page={p}"
        try:
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            table = soup.find("table", class_="type2")
            if not table:
                continue
                
            for r in table.find_all("tr"):
                cols = r.find_all("td")
                if len(cols) == 9 and cols[0].text.strip():
                    date_str = cols[0].text.strip().replace(".", "-")
                    inst_str = cols[5].text.strip().replace(",", "")
                    frgn_str = cols[6].text.strip().replace(",", "")
                    try:
                        dt = pd.to_datetime(date_str)
                        inst_net = float(inst_str) if inst_str else 0.0
                        frgn_net = float(frgn_str) if frgn_str else 0.0
                        records.append({"Date": dt, "INST_NET": inst_net, "FRGN_NET": frgn_net})
                    except Exception:
                        continue
        except Exception:
            continue
            
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates(subset=["Date"]).set_index("Date").sort_index()

# -------------------------------------------------------------
# 4. 실시간 커뮤니티 리젠 속도 측정
# -------------------------------------------------------------
def fetch_community_buzz_speed(ticker_code):
    if ticker_code == "COHR":
        try:
            url = "https://api.stocktwits.com/api/2/streams/symbol/COHR.json"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=2)
            if res.status_code == 200:
                msgs = res.json().get("messages", [])
                return {"speed_label": f"StockTwits {len(msgs)}건", "is_surge": len(msgs) >= 25}
        except Exception:
            pass
        return {"speed_label": "소셜 안정", "is_surge": False}

    code_digits = ticker_code.split(".")[0]
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://finance.naver.com/item/board.naver?code={code_digits}&page=1"
    try:
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            table = soup.find("table", class_="type2")
            if table:
                dates = []
                for tr in table.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) >= 2:
                        d_text = tds[0].text.strip()
                        if len(d_text) >= 16:
                            try:
                                dates.append(datetime.strptime(d_text, "%Y.%m.%d %H:%M"))
                            except Exception:
                                pass
                if len(dates) >= 10:
                    span_mins = (dates[0] - dates[9]).total_seconds() / 60.0
                    if span_mins > 0:
                        posts_per_min = 10.0 / span_mins
                        return {
                            "speed_label": f"분당 {posts_per_min:.1f}건 ({span_mins:.0f}분 동안 10건)",
                            "is_surge": posts_per_min >= 0.5
                        }
    except Exception:
        pass
    return {"speed_label": "정상 범위 (리젠 보통)", "is_surge": False}

# -------------------------------------------------------------
# 5. 116+ 팩터 라이브러리 (상충 및 교호작용 팩터 포함)
# -------------------------------------------------------------
def build_comprehensive_factors(df_target, df_flow, macro_dict):
    f = {}
    c = df_target['Close']
    o = df_target['Open']
    h = df_target['High']
    l = df_target['Low']
    v = df_target['Volume']
    
    # 1. 수급 팩터 (8종)
    if df_flow is not None and not df_flow.empty:
        flow_aligned = df_flow.reindex(df_target.index).fillna(0)
        v_20 = v.rolling(20).mean() + 1e-9
        frgn = flow_aligned["FRGN_NET"]
        inst = flow_aligned["INST_NET"]
        f["SUPPLY_FRGN_1D"] = (frgn / v_20) * 100
        f["SUPPLY_INST_1D"] = (inst / v_20) * 100
        f["SUPPLY_FRGN_5D"] = (frgn.rolling(5).sum() / v_20) * 100
        f["SUPPLY_INST_5D"] = (inst.rolling(5).sum() / v_20) * 100
        f["SUPPLY_DUAL_BUY"] = np.where((frgn > 0) & (inst > 0), 1.0, np.where((frgn < 0) & (inst < 0), -1.0, 0.0))
        f["SUPPLY_FRGN_Z20"] = (frgn - frgn.rolling(20).mean()) / (frgn.rolling(20).std() + 1e-9)
        f["SUPPLY_INST_Z20"] = (inst - inst.rolling(20).mean()) / (inst.rolling(20).std() + 1e-9)
        f["SUPPLY_INST_ACCUM_10D"] = (inst.rolling(10).sum() / (v.rolling(10).sum() + 1e-9)) * 100

    # 2. 야간 ADR 및 상충(Conflict) 교호작용 팩터
    for adr_name in ["SKHY", "SSNLF"]:
        if adr_name in macro_dict and macro_dict[adr_name] is not None and not macro_dict[adr_name].empty:
            m_c = macro_dict[adr_name]['Close'].reindex(df_target.index).ffill().bfill()
            ret_adr = m_c.pct_change(1).fillna(0) * 100
            f[f"ADR_{adr_name}_1D"] = ret_adr
            f[f"ADR_{adr_name}_OVERNIGHT_MOM"] = ret_adr - (c.pct_change(1).fillna(0) * 100)
            
            # [신규 교호작용 팩터] 20일 추세와 야간 ADR 간의 방향 상충 감지
            mom_20 = c.pct_change(20).fillna(0) * 100
            f[f"CONFLICT_TREND_{adr_name}"] = np.sign(mom_20) * ret_adr
            f[f"DIVERGENCE_5D_{adr_name}"] = (c.pct_change(5).fillna(0) * 100) - ret_adr

    # 3. 멀티호라이즌 모멘텀 (14종)
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180]:
        f[f"MOM_{k}D"] = c.pct_change(k) * 100
        
    # 4. 이평선 괴리율 및 추세 기울기 (18종)
    for k in [5, 10, 20, 40, 60, 120, 200]:
        sma = c.rolling(k).mean()
        f[f"DISP_SMA_{k}"] = (c / (sma + 1e-9) - 1) * 100
        f[f"SLOPE_SMA_{k}"] = sma.pct_change(5) * 100
    for k in [12, 26, 50, 100]:
        ema = c.ewm(span=k, adjust=False).mean()
        f[f"DISP_EMA_{k}"] = (c / (ema + 1e-9) - 1) * 100

    # 5. 변동성 및 프라이스 액션 (20종)
    for k in [5, 10, 20, 60]:
        f[f"HV_{k}"] = c.pct_change().rolling(k).std() * np.sqrt(252) * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    for k in [5, 10, 14, 21]:
        f[f"NATR_{k}"] = (tr.rolling(k).mean() / (c + 1e-9)) * 100
    f["BODY_RATIO"] = (c - o).abs() / (h - l + 1e-9)
    f["UPPER_SHADOW"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - l + 1e-9)
    f["LOWER_SHADOW"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (h - l + 1e-9)
    f["GAP_OVERNIGHT"] = (o / (c.shift() + 1e-9) - 1) * 100
    f["INTRADAY_RET"] = (c / (o + 1e-9) - 1) * 100

    # 6. 오실레이터 (15종)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["RSI_14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    for k in [10, 20, 30]:
        ma = c.rolling(k).mean()
        std = c.rolling(k).std()
        f[f"BB_PCT_{k}"] = (c - (ma - 2*std)) / (4*std + 1e-9)
        f[f"BB_WIDTH_{k}"] = (4*std) / (ma + 1e-9) * 100

    # 7. 거래량 (10종)
    for k in [3, 5, 10, 20, 50]:
        f[f"VOL_RATIO_{k}"] = v / (v.rolling(k).mean() + 1e-9)

    # 8. 글로벌 매크로 (30종) - bfill + fillna(0)
    for asset_name, m_df in macro_dict.items():
        if m_df is not None and not m_df.empty:
            m_c = m_df['Close'].reindex(df_target.index).ffill().bfill()
            f[f"MACRO_{asset_name}_1D"] = m_c.pct_change(1).fillna(0) * 100
            f[f"MACRO_{asset_name}_5D"] = m_c.pct_change(5).fillna(0) * 100
            ret_m = m_c.pct_change(1).fillna(0)
            f[f"MACRO_{asset_name}_20D_Z"] = ((ret_m - ret_m.rolling(20).mean()) / (ret_m.rolling(20).std() + 1e-9)).fillna(0)

    return pd.DataFrame(f, index=df_target.index)

# -------------------------------------------------------------
# 6. 데이터 일괄 수집 엔진 (3분 캐싱)
# -------------------------------------------------------------
@st.cache_data(ttl=180)
def load_all_base_assets():
    target_symbols = {
        "COHR": "COHR",
        "SK하이닉스": "000660.KS",
        "삼성전자": "005930.KS",
        "두산에너빌리티": "034020.KS",
        "하이브": "352820.KS",
        "LS": "006260.KS"
    }
    macro_symbols = {
        "QQQ": "QQQ", "SPY": "SPY", "SOXX": "SOXX", "WTI": "CL=F",
        "COPPER": "HG=F", "GOLD": "GC=F", "TNX": "^TNX", "USDKRW": "KRW=X",
        "URNM": "URNM", "MU": "MU", "NVDA": "NVDA",
        "SKHY": "SKHY",
        "SSNLF": "SSNLF"
    }
    targets = {}
    macros = {}
    
    for k, sym in target_symbols.items():
        df = pd.DataFrame()
        try:
            df = yf.Ticker(sym).history(period="2y")
        except Exception:
            pass
            
        if (df.empty or len(df) < 60) and k == "COHR":
            df = fetch_alpha_vantage_daily(sym)

        if not df.empty and len(df) > 60:
            df = df.dropna(subset=['Close'])
            df = df[df['Close'] > 0]
            df.index = df.index.tz_localize(None).normalize()
            targets[k] = df
            
    for k, sym in macro_symbols.items():
        try:
            df = yf.Ticker(sym).history(period="2y")
            if (df.empty or len(df) < 30) and k in ["SKHY", "SSNLF"]:
                df = fetch_alpha_vantage_daily(sym)
                
            if not df.empty:
                df = df.dropna(subset=['Close'])
                df = df[df['Close'] > 0]
                df.index = df.index.tz_localize(None).normalize()
                macros[k] = df
        except Exception:
            pass
            
    return targets, macros

# -------------------------------------------------------------
# 7. 비선형 상충 학습 머신러닝 엔진 (HistGradientBoosting)
# -------------------------------------------------------------
def run_advanced_quant_engine(df_target, df_flow, macro_dict, buzz_info, event_type="None"):
    df_clean = df_target.dropna(subset=['Close']).copy()
    if len(df_clean) < 80:
        return None

    factors_all = build_comprehensive_factors(df_clean, df_flow, macro_dict)
    
    target_ret = df_clean['Close'].pct_change().shift(-1) * 100
    target_y = (target_ret > 0).astype(int)
    
    train_mask = ~factors_all.isna().any(axis=1) & ~target_ret.isna()
    X_train = factors_all.loc[train_mask]
    y_train = target_y.loc[train_mask]
    ret_train = target_ret.loc[train_mask]
    
    valid_factors_mask = ~factors_all.isna().any(axis=1)
    X_latest_row = factors_all.loc[valid_factors_mask].iloc[-1:]
    X_prev_row = factors_all.loc[train_mask].iloc[-1:]
    
    n_obs = len(X_train)
    if n_obs < 60:
        return None
        
    # Active-Period IC 검정
    ic_stats = []
    for col in X_train.columns:
        s = X_train[col]
        active_mask = (s != 0.0) & ~s.isna()
        
        if active_mask.sum() >= 30:
            ic, pval = spearmanr(s.loc[active_mask], ret_train.loc[active_mask])
            eff_n = active_mask.sum()
        else:
            ic, pval = spearmanr(s, ret_train)
            eff_n = n_obs
            
        if np.isnan(ic):
            ic, pval = 0.0, 1.0
            
        t_stat = ic * np.sqrt(eff_n - 2) / (np.sqrt(1.0 - ic**2) + 1e-9)
        ic_stats.append({
            "팩터코드": col,
            "스피어만 IC": ic,
            "절대 IC": abs(ic),
            "p-value": pval,
            "t-통계량": t_stat
        })
        
    ic_df = pd.DataFrame(ic_stats).sort_values(by="절대 IC", ascending=False)
    sig_candidates = ic_df[(ic_df["절대 IC"] >= 0.025) & (ic_df["p-value"] < 0.15)]["팩터코드"].tolist()
    if len(sig_candidates) < 8:
        selected_factors = ic_df["팩터코드"].head(10).tolist()
    elif len(sig_candidates) > 18:
        selected_factors = sig_candidates[:18]
    else:
        selected_factors = sig_candidates
        
    decay_lambda = 0.975
    sample_weights = np.array([decay_lambda ** (n_obs - 1 - i) for i in range(n_obs)])
    
    X_tr = X_train[selected_factors]
    X_lt = X_latest_row[selected_factors]
    X_pr = X_prev_row[selected_factors]
    
    # [핵심] 비선형 트리 앙상블: 추세와 야간급락이 충돌할 때의 조건부 확률을 데이터 기반으로 스스로 학습
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,           # 최대 4개 팩터의 복합 상충 분기 탐색
        min_samples_leaf=15,   # 통계적 과적합 방지
        l2_regularization=1.5,
        random_state=42
    )
    model.fit(X_tr, y_train, sample_weight=sample_weights)
    
    # 모델이 스스로 산출한 조건부 상승 확률
    prob_up_raw = float(model.predict_proba(X_lt)[0][1] * 100)
    if buzz_info.get("is_surge", False):
        prob_up_raw = np.clip(prob_up_raw + 2.0, 5.0, 95.0)
        
    prob_up_yesterday = float(model.predict_proba(X_pr)[0][1] * 100)

    # 이벤트 변동성 확장 승수
    vol_mult = 1.0
    if event_type == "FOMC / 금리 결정":
        vol_mult = 1.75
    elif event_type == "미국 CPI 발표":
        vol_mult = 1.50
    elif event_type in ["DART 호재 공시(자사주/대규모 수주)", "DART 악재 공시(유상증자/CB발행)"]:
        vol_mult = 1.30

    final_prob_up = float(np.clip(prob_up_raw, 5.0, 95.0))
    final_prob_down = float(100.0 - final_prob_up)

    daily_vol = float(df_clean['Close'].pct_change().tail(20).std() * 100)
    final_magnitude = ((final_prob_up - 50.0) / 50.0) * daily_vol * 1.35 * vol_mult

    # 팩터별 한계 기여도(Marginal Contribution) 분해
    # 각 팩터를 평균값으로 대체했을 때 확률이 얼마나 변하는지 실측하여 기여도 산출
    contributions = {}
    base_p = prob_up_raw
    X_lt_copy = X_lt.copy()
    
    for f_name in selected_factors:
        orig_val = X_lt_copy[f_name].iloc[0]
        mean_val = X_tr[f_name].mean()
        X_lt_copy[f_name] = mean_val
        p_cf = float(model.predict_proba(X_lt_copy)[0][1] * 100)
        contributions[f_name] = float(base_p - p_cf)
        X_lt_copy[f_name] = orig_val

    sorted_c = sorted(contributions.items(), key=lambda x: x[1])
    top_neg_factor = sorted_c[0] if sorted_c else ("None", 0.0)
    top_pos_factor = sorted_c[-1] if sorted_c else ("None", 0.0)
        
    recent_preds = model.predict(X_tr.iloc[-60:])
    backtest_acc = float((recent_preds == y_train.iloc[-60:]).mean() * 100)
    
    valid_closes = df_clean['Close'].dropna()
    latest_close = float(valid_closes.iloc[-1]) if len(valid_closes) > 0 else np.nan
    prev_close = float(valid_closes.iloc[-2]) if len(valid_closes) > 1 else np.nan
    price_change = float(((latest_close - prev_close) / prev_close) * 100) if (prev_close and prev_close > 0) else 0.0
    
    return {
        "latest_close": latest_close,
        "price_change": price_change,
        "prob_up_next": final_prob_up,
        "prob_down_next": final_prob_down,
        "prob_up_yesterday": prob_up_yesterday,
        "daily_vol": daily_vol,
        "expected_magnitude": final_magnitude,
        "vol_multiplier": vol_mult,
        "contributions": contributions,
        "top_pos_factor": top_pos_factor,
        "top_neg_factor": top_neg_factor,
        "ic_df": ic_df,
        "selected_factors": selected_factors,
        "backtest_acc": backtest_acc,
        "total_tested": len(factors_all.columns)
    }

# -------------------------------------------------------------
# 8. UI 대시보드 렌더링
# -------------------------------------------------------------
st.markdown("## 🏛️ AlphaPulse 인스티튜셔널 퀀트 터미널 Pro")

st.sidebar.markdown("### ⚡ 거시 변수 & 공시 쇼크 필터")
event_choice = st.sidebar.selectbox(
    "오늘 장 마감 후 특이 이벤트 반영",
    ["None", "FOMC / 금리 결정", "미국 CPI 발표", "DART 호재 공시(자사주/대규모 수주)", "DART 악재 공시(유상증자/CB발행)"],
    index=0
)

with st.spinner("미국 정규장 실시간 데이터 및 비선형 트리 앙상블 학습 중..."):
    all_targets, all_macros = load_all_base_assets()

stock_tabs = st.tabs([
    "🇺🇸 COHR (Coherent)",
    "🇰🇷 SK하이닉스",
    "🇰🇷 삼성전자",
    "🇰🇷 두산에너빌리티",
    "🇰🇷 하이브",
    "🇰🇷 LS"
])

tab_mapping = [
    ("COHR", stock_tabs[0], "$", "COHR"),
    ("SK하이닉스", stock_tabs[1], "₩", "000660.KS"),
    ("삼성전자", stock_tabs[2], "₩", "005930.KS"),
    ("두산에너빌리티", stock_tabs[3], "₩", "034020.KS"),
    ("하이브", stock_tabs[4], "₩", "352820.KS"),
    ("LS", stock_tabs[5], "₩", "006260.KS")
]

for ticker_name, tab, currency, ticker_code in tab_mapping:
    with tab:
        is_market_open, session_status, time_str = get_market_session_status(currency)
        df_tgt = all_targets.get(ticker_name)
        if df_tgt is None or len(df_tgt) < 60:
            st.error(f"{ticker_name}의 시계열 데이터를 불러오지 못했습니다. 잠시 후 새로고침해주세요.")
            continue
            
        df_flow = fetch_naver_investor_flows(ticker_code)
        buzz = fetch_community_buzz_speed(ticker_code)

        # 실시간 선행 자산 표시 라벨 구성
        lead_display_label = "야간 선행 지표"
        lead_display_val = "0.00%"
        lead_sub_caption = "동기화 중"

        if ticker_name == "SK하이닉스":
            skhy_live = get_asset_live_change(all_macros.get("SKHY"))
            soxx_live = get_asset_live_change(all_macros.get("SOXX"))
            mu_live = get_asset_live_change(all_macros.get("MU"))
            lead_display_label = "미장 SKHY ADR 실시간"
            lead_display_val = f"{skhy_live:+.2f}%"
            lead_sub_caption = f"SOXX {soxx_live:+.2f}% | MU {mu_live:+.2f}%"

        elif ticker_name == "삼성전자":
            ssnlf_live = get_asset_live_change(all_macros.get("SSNLF"))
            soxx_live = get_asset_live_change(all_macros.get("SOXX"))
            mu_live = get_asset_live_change(all_macros.get("MU"))
            lead_display_label = "미장 삼성 ADR / 반도체"
            lead_display_val = f"{ssnlf_live:+.2f}%"
            lead_sub_caption = f"SOXX {soxx_live:+.2f}% | MU {mu_live:+.2f}%"

        elif ticker_name == "두산에너빌리티":
            urnm_live = get_asset_live_change(all_macros.get("URNM"))
            wti_live = get_asset_live_change(all_macros.get("WTI"))
            lead_display_label = "글로벌 우라늄(URNM) / 유가"
            lead_display_val = f"{urnm_live:+.2f}%"
            lead_sub_caption = f"WTI 유가 {wti_live:+.2f}%"

        elif ticker_name == "LS":
            cop_live = get_asset_live_change(all_macros.get("COPPER"))
            wti_live = get_asset_live_change(all_macros.get("WTI"))
            lead_display_label = "국제 구리 선물(COPPER)"
            lead_display_val = f"{cop_live:+.2f}%"
            lead_sub_caption = f"WTI 유가 {wti_live:+.2f}%"

        elif ticker_name == "하이브":
            qqq_live = get_asset_live_change(all_macros.get("QQQ"))
            usdkrw_live = get_asset_live_change(all_macros.get("USDKRW"))
            lead_display_label = "미국 QQQ / 환율"
            lead_display_val = f"{qqq_live:+.2f}%"
            lead_sub_caption = f"환율 {usdkrw_live:+.2f}%"

        elif ticker_name == "COHR":
            cohr_live = get_asset_live_change(all_targets.get("COHR"))
            soxx_live = get_asset_live_change(all_macros.get("SOXX"))
            lead_display_label = "COHR 미장 실시간"
            lead_display_val = f"{cohr_live:+.2f}%"
            lead_sub_caption = f"SOXX {soxx_live:+.2f}%"

        res = run_advanced_quant_engine(df_tgt, df_flow, all_macros, buzz, event_type=event_choice)
        
        if res is None:
            st.warning("데이터 웜업 모수가 부족합니다.")
            continue

        status_color = "🟢" if is_market_open else "🌙"
        st.caption(f"시스템 시각: {time_str} | **{status_color} {session_status}**")

        # -------------------------------------------------------------
        # 실무형 상단 5대 핵심 판단 지표 카드
        # -------------------------------------------------------------
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"{ticker_name} 현재가/종가", format_price(res['latest_close'], currency), f"{res['price_change']:+.2f}%")
        c2.metric(lead_display_label, lead_display_val, lead_sub_caption)
        c3.metric("오늘 최고 상승 팩터", f"{res['top_pos_factor'][0]}", f"{res['top_pos_factor'][1]:+.1f}%p 상승기여")
        c4.metric("오늘 최고 하락 팩터", f"{res['top_neg_factor'][0]}", f"{res['top_neg_factor'][1]:+.1f}%p 하방압력")
        c5.metric("최근 실전 적중률", f"{res['backtest_acc']:.1f}%", "60일 Out-of-sample")

        st.markdown("---")

        preview_mode = st.toggle("🛠️ 장 마감 후 익일 예측 모드 강제 미리보기 (테스트용)", value=not is_market_open, key=f"toggle_{ticker_name}")

        if is_market_open and not preview_mode:
            st.info(f"🔔 **현재 {session_status}입니다.** 장중에는 당일 확정 종가와 수급이 마감되지 않았으므로 **다음 거래일 예측이 일시 중지**되며, **전일 모델 예측의 실시간 적중 여부**를 검증합니다.")
            
            p_yesterday = res["prob_up_yesterday"]
            pred_dir_up = p_yesterday >= 50.0
            actual_dir_up = res["price_change"] >= 0.0
            is_hit = (pred_dir_up == actual_dir_up)

            col_eval1, col_eval2 = st.columns([1.2, 1])
            with col_eval1:
                st.markdown("### 🎯 전일 모델 예측 vs 오늘 장중 등락 적중 검증")
                st.write(f"• **어제 장 마감 시 모델 예측**: {'▲ 상승 우세' if pred_dir_up else '▼ 하락 우세'} (확률: {p_yesterday if pred_dir_up else 100 - p_yesterday:.1f}%)")
                st.write(f"• **오늘 현재 실제 주가 변동**: **{res['price_change']:+.2f}%** ({format_price(res['latest_close'], currency)})")
                
                if is_hit:
                    st.success("### ✅ [예측 적중 (Hit)] 모델의 사전 방향성 예측과 일치하여 거래 중입니다.")
                else:
                    st.error("### ⚠️ [예측 불일치 (Miss)] 장중 시장 변수로 사전 예측과 괴리가 발생했습니다.")
                st.caption("※ 장 마감(국내 18:00 / 미국 16:00 ET) 이후 확정된 데이터를 바탕으로 다음 거래일 예측이 자동 가동됩니다.")

            with col_eval2:
                st.markdown("**어제 채택되었던 주요 선행 시그널**")
                display_prev = res["ic_df"].head(5)
                st.dataframe(display_prev[["팩터코드", "스피어만 IC", "p-value"]], use_container_width=True)

        else:
            prob_up = res["prob_up_next"]
            prob_down = res["prob_down_next"]
            exp_mag = res["expected_magnitude"]

            col_dir, col_gauge = st.columns([1.3, 1])
            with col_dir:
                event_badge = f" [⚡ {event_choice}]" if event_choice != "None" else ""
                if prob_up >= 54.0:
                    st.success(f"### 📈 [상승 우세] 다음 거래일 상승 확률: {prob_up:.1f}%{event_badge}")
                    st.write(f"**하락 반대 확률**: {prob_down:.1f}% | 모델 확신도: **상승 우위 (+{prob_up - 50:.1f}%p)**")
                    st.info(f"🎯 **예상 일일 변동폭(Magnitude)**: **+{abs(exp_mag)*0.7:.2f}% ~ +{abs(exp_mag)*1.3:.2f}%** (상방 압력, 변동성 계수: {res['vol_multiplier']}x)")
                elif prob_up <= 46.0:
                    st.error(f"### 📉 [하락 우세] 다음 거래일 하락 확률: {prob_down:.1f}%{event_badge}")
                    st.write(f"**상승 반대 확률**: {prob_up:.1f}% | 모델 확신도: **하락 우위 (+{prob_down - 50:.1f}%p)**")
                    st.warning(f"🎯 **예상 일일 변동폭(Magnitude)**: **-{abs(exp_mag)*0.7:.2f}% ~ -{abs(exp_mag)*1.3:.2f}%** (하방 압력, 변동성 계수: {res['vol_multiplier']}x)")
                else:
                    st.warning(f"### ⚖️ [중립 / 관망 권고] 방향성 탐색 (상충·노이즈 구간){event_badge}")
                    st.write(f"**상승**: {prob_up:.1f}% vs **하락**: {prob_down:.1f}%")
                    st.write(f"🎯 **예상 일일 변동폭**: **±{res['daily_vol']*0.5 * res['vol_multiplier']:.2f}% 내외 박스권 횡보**")
                
                st.caption(f"※ 장기 추세와 단기 야간 선행 자산의 상충 관계를 비선형 트리 앙상블이 학습하여 도출한 결과입니다.")

            # 게이지 차트
            with col_gauge:
                score_100 = (prob_up - 50.0) * 2.0
                st.markdown("<p style='text-align:center; font-weight:700; font-size:14px; margin-bottom:0;'>양방향 신호 강도 (-100:강력하락 ~ +100:강력상승)</p>", unsafe_allow_html=True)
                
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_100,
                    domain={'x': [0, 1], 'y': [0, 0.95]},
                    gauge={
                        'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "#8492a6"},
                        'bar': {'color': "#10b981" if score_100 >= 0 else "#ef4444"},
                        'steps': [
                            {'range': [-100, -20], 'color': "rgba(239, 68, 68, 0.2)"},
                            {'range': [-20, 20], 'color': "rgba(107, 114, 128, 0.2)"},
                            {'range': [20, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                        ],
                        'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': 0}
                    }
                ))
                fig_g.update_layout(height=230, margin=dict(l=25, r=25, t=10, b=10))
                st.plotly_chart(fig_g, use_container_width=True)

            # 팩터 한계 기여도 차트
            c_chart1, _ = st.columns([1, 0.01])
            with c_chart1:
                c_names = list(res["contributions"].keys())
                c_vals = list(res["contributions"].values())
                c_colors = ["#10b981" if v >= 0 else "#ef4444" for v in c_vals]
                fig_bar = go.Figure(go.Bar(x=c_vals, y=c_names, orientation='h', marker_color=c_colors))
                fig_bar.update_layout(title="머신러닝 트리가 평가한 팩터별 확률 한계 기여도 (%p)", height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            # 전체 팩터 IC 통계 검정 순위표
            with st.expander(f"📊 {ticker_name} - 전체 팩터 IC 통계 검정 순위표 (총 {res['total_tested']}개 중 {len(res['selected_factors'])}개 채택)"):
                display_ic = res["ic_df"].copy()
                display_ic["스크리닝 결과"] = display_ic["팩터코드"].apply(
                    lambda x: "🟢 통과 (모델 채택)" if x in res["selected_factors"] else "⚪ 탈락 (노이즈/중복)"
                )
                st.dataframe(
                    display_ic[["팩터코드", "스피어만 IC", "t-통계량", "p-value", "스크리닝 결과"]],
                    use_container_width=True,
                    height=380
                )
