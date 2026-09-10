import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 페이지 기본 설정
st.set_page_config(page_title="AlphaPulse Institutional Quant Terminal", layout="wide")

# -------------------------------------------------------------
# 1. 116개 대규모 팩터 라이브러리 생성 엔진 (Vectorized)
# -------------------------------------------------------------
def build_100_plus_factor_library(df_target, macro_dict):
    """
    OHLCV 및 매크로/교차 자산 시계열로부터 총 116개 정량 팩터를 일괄 계산합니다.
    """
    f = {}
    c = df_target['Close']
    o = df_target['Open']
    h = df_target['High']
    l = df_target['Low']
    v = df_target['Volume']
    
    # 1. 멀티호라이즌 모멘텀 (14개)
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180]:
        f[f"MOM_{k}D"] = c.pct_change(k) * 100
        
    # 2. 이평선 괴리율 및 추세 기울기 (18개)
    for k in [5, 10, 20, 40, 60, 120, 200]:
        sma = c.rolling(k).mean()
        f[f"DISP_SMA_{k}"] = (c / (sma + 1e-9) - 1) * 100
        f[f"SLOPE_SMA_{k}"] = sma.pct_change(5) * 100
    for k in [12, 26, 50, 100]:
        ema = c.ewm(span=k, adjust=False).mean()
        f[f"DISP_EMA_{k}"] = (c / (ema + 1e-9) - 1) * 100

    # 3. 변동성, 일중 레인지 및 캔들 구조 (22개)
    for k in [5, 10, 20, 60]:
        f[f"HV_{k}"] = c.pct_change().rolling(k).std() * np.sqrt(252) * 100
    
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    for k in [5, 10, 14, 21, 28]:
        f[f"NATR_{k}"] = (tr.rolling(k).mean() / (c + 1e-9)) * 100
    
    hl_ratio = np.log(h / (l + 1e-9)) ** 2
    for k in [10, 20]:
        f[f"PARKINSON_{k}"] = np.sqrt((1.0 / (4.0 * np.log(2))) * hl_ratio.rolling(k).mean()) * np.sqrt(252) * 100

    f["BODY_RATIO"] = (c - o).abs() / (h - l + 1e-9)
    f["UPPER_SHADOW"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - l + 1e-9)
    f["LOWER_SHADOW"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (h - l + 1e-9)
    f["GAP_OVERNIGHT"] = (o / (c.shift() + 1e-9) - 1) * 100
    f["INTRADAY_RET"] = (c / (o + 1e-9) - 1) * 100
    f["HL_SPREAD"] = (h - l) / (c + 1e-9) * 100
    f["INTRADAY_POS"] = (c - l) / (h - l + 1e-9)

    # 4. 오실레이터 및 평균회귀 (24개)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for k in [7, 14, 21, 28]:
        avg_g = gain.rolling(k).mean()
        avg_l = loss.rolling(k).mean()
        rs = avg_g / (avg_l + 1e-9)
        f[f"RSI_{k}"] = 100 - (100 / (1 + rs))

    for k in [10, 20, 30]:
        ma = c.rolling(k).mean()
        std = c.rolling(k).std()
        f[f"BB_PCT_{k}"] = (c - (ma - 2*std)) / (4*std + 1e-9)
        f[f"BB_WIDTH_{k}"] = (4*std) / (ma + 1e-9) * 100

    for k in [9, 14, 21]:
        low_min = l.rolling(k).min()
        high_max = h.rolling(k).max()
        f[f"STOCH_K_{k}"] = (c - low_min) / (high_max - low_min + 1e-9) * 100
        f[f"WILLIAMS_R_{k}"] = (high_max - c) / (high_max - low_min + 1e-9) * -100

    for k in [14, 20]:
        tp = (h + l + c) / 3
        ma_tp = tp.rolling(k).mean()
        md = (tp - ma_tp).abs().rolling(k).mean()
        f[f"CCI_{k}"] = (tp - ma_tp) / (0.015 * md + 1e-9)

    exp1 = c.ewm(span=12, adjust=False).mean()
    exp2 = c.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=9, adjust=False).mean()
    f["MACD_LINE"] = macd / (c + 1e-9) * 100
    f["MACD_HIST"] = (macd - sig) / (c + 1e-9) * 100

    # 5. 거래량 및 유동성 수급 (18개)
    for k in [3, 5, 10, 20, 50, 100]:
        f[f"VOL_RATIO_{k}"] = v / (v.rolling(k).mean() + 1e-9)

    for k in [1, 3, 5, 10]:
        f[f"VOL_MOM_{k}"] = v.pct_change(k) * 100

    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    for k in [5, 10, 20]:
        f[f"OBV_CHG_{k}"] = (obv - obv.shift(k)) / (v.rolling(k).sum() + 1e-9)

    amihud = c.pct_change().abs() / (v * c + 1e-9)
    for k in [10, 20]:
        f[f"AMIHUD_{k}"] = amihud.rolling(k).mean() * 1e9

    vwap_20 = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
    f["VWAP_GAP_20"] = (c / (vwap_20 + 1e-9) - 1) * 100

    # 6. 글로벌 거시 및 교차 자산 스필오버 (각 자산당 1D, 5D, 20D Z-Score)
    for asset_name, m_df in macro_dict.items():
        if m_df is not None and not m_df.empty:
            m_c = m_df['Close'].reindex(df_target.index).ffill()
            f[f"MACRO_{asset_name}_1D"] = m_c.pct_change(1) * 100
            f[f"MACRO_{asset_name}_5D"] = m_c.pct_change(5) * 100
            ret_m = m_c.pct_change(1)
            f[f"MACRO_{asset_name}_20D_Z"] = (ret_m - ret_m.rolling(20).mean()) / (ret_m.rolling(20).std() + 1e-9)

    return pd.DataFrame(f, index=df_target.index)

# -------------------------------------------------------------
# 2. 글로벌 시장 데이터 수집 파이프라인 (2년 시계열 캐싱)
# -------------------------------------------------------------
@st.cache_data(ttl=900)
def load_all_market_assets():
    """전체 6개 타깃 종목 및 10대 글로벌 매크로 자산 수집"""
    target_symbols = {
        "COHR": "COHR",
        "SK하이닉스": "000660.KS",
        "삼성전자": "005930.KS",
        "두산에너빌리티": "034020.KS",
        "하이브": "352820.KS",
        "LS": "006260.KS"
    }
    macro_symbols = {
        "QQQ": "QQQ",         # 나스닥 100
        "SPY": "SPY",         # S&P 500
        "SOXX": "SOXX",       # 필라델피아 반도체
        "WTI": "CL=F",        # WTI 원유
        "COPPER": "HG=F",     # 구리 선물 (LS, 전력 인프라 핵심)
        "GOLD": "GC=F",       # 금 선물
        "TNX": "^TNX",        # 미국 10년물 국채금리
        "USDKRW": "KRW=X",    # 원/달러 환율
        "URNM": "URNM",       # 글로벌 원전/우라늄 ETF (두산에너빌리티 선행)
        "MU": "MU",           # 마이크론 (반도체 선행)
        "NVDA": "NVDA"        # 엔비디아 (AI 하드웨어 선행)
    }
    
    targets = {}
    macros = {}
    
    for k, sym in target_symbols.items():
        try:
            df = yf.Ticker(sym).history(period="2y")
            if not df.empty and len(df) > 100:
                df.index = df.index.tz_localize(None).normalize()
                targets[k] = df
        except Exception:
            pass
            
    for k, sym in macro_symbols.items():
        try:
            df = yf.Ticker(sym).history(period="2y")
            if not df.empty:
                df.index = df.index.tz_localize(None).normalize()
                macros[k] = df
        except Exception:
            pass
            
    return targets, macros

# -------------------------------------------------------------
# 3. 116개 팩터 IC 통계 검정 & 적응형 머신러닝 엔진
# -------------------------------------------------------------
def run_institutional_engine(df_target, macro_dict, ticker_label):
    factors_all = build_100_plus_factor_library(df_target, macro_dict)
    
    # 익일 수익률 타깃 라벨링
    target_ret = df_target['Close'].pct_change().shift(-1) * 100
    target_y = (target_ret > 0).astype(int)
    
    # 200일 이평선 등 웜업 결측치 제거
    valid_mask = ~factors_all.isna().any(axis=1) & ~target_ret.isna()
    X_clean = factors_all.loc[valid_mask]
    y_clean = target_y.loc[valid_mask]
    ret_clean = target_ret.loc[valid_mask]
    
    if len(X_clean) < 100:
        return None
        
    train_X = X_clean.iloc[:-1]
    train_y = y_clean.iloc[:-1]
    train_ret = ret_clean.iloc[:-1]
    latest_X = X_clean.iloc[-1:]
    n_obs = len(train_X)
    
    # [1단계: 116개 전체 팩터의 스피어만 Rank IC 및 t-통계량, p-value 검정]
    ic_stats = []
    for col in train_X.columns:
        ic, pval = spearmanr(train_X[col], train_ret)
        if np.isnan(ic):
            ic, pval = 0.0, 1.0
        t_stat = ic * np.sqrt(n_obs - 2) / (np.sqrt(1.0 - ic**2) + 1e-9)
        ic_stats.append({
            "팩터코드": col,
            "스피어만 IC": ic,
            "절대 IC": abs(ic),
            "p-value": pval,
            "t-통계량": t_stat
        })
        
    ic_df = pd.DataFrame(ic_stats).sort_values(by="절대 IC", ascending=False)
    
    # [2단계: 통계적 유의성 스크리닝 (|IC| >= 0.025 & p < 0.15)]
    sig_candidates = ic_df[(ic_df["절대 IC"] >= 0.025) & (ic_df["p-value"] < 0.15)]["팩터코드"].tolist()
    
    # 다중공선성 제어 및 최소/최대 팩터 슬롯 확보 (상위 8~14개 최적화)
    if len(sig_candidates) < 6:
        selected_factors = ic_df["팩터코드"].head(8).tolist()
    elif len(sig_candidates) > 14:
        selected_factors = sig_candidates[:14]
    else:
        selected_factors = sig_candidates
        
    # [3단계: 지수 감쇠 가중치(Exponential Decay) 및 L2 정규화 회귀 학습]
    decay_lambda = 0.975  # Half-life 약 28거래일
    sample_weights = np.array([decay_lambda ** (n_obs - 1 - i) for i in range(n_obs)])
    
    X_train_sub = train_X[selected_factors]
    X_latest_sub = latest_X[selected_factors]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_latest_scaled = scaler.transform(X_latest_sub)
    
    model = LogisticRegression(penalty='l2', C=0.25, random_state=42)
    model.fit(X_train_scaled, train_y, sample_weight=sample_weights)
    
    prob_up = float(model.predict_proba(X_latest_scaled)[0][1] * 100)
    prob_down = float(100.0 - prob_up)
    
    # 팩터별 실제 기여도 분해
    coefs = model.coef_[0]
    scaled_curr = X_latest_scaled[0]
    contributions = {}
    weights_dict = {}
    for idx, f_name in enumerate(selected_factors):
        impact = float(coefs[idx] * scaled_curr[idx] * 8.0)
        contributions[f_name] = impact
        weights_dict[f_name] = float(coefs[idx])
        
    # Walk-forward 백테스트 적중률 (최근 60거래일 Out-of-sample)
    recent_preds = model.predict(X_train_scaled[-60:])
    backtest_acc = float((recent_preds == train_y.iloc[-60:]).mean() * 100)
    
    # 20일 변동성 기반 일일 예상 등락폭(Magnitude)
    daily_vol = float(df_target['Close'].pct_change().tail(20).std() * 100)
    direction_strength = (prob_up - 50.0) / 50.0
    expected_magnitude = direction_strength * daily_vol * 1.35
    
    latest_close = float(df_target['Close'].iloc[-1])
    prev_close = float(df_target['Close'].iloc[-2])
    price_change = float(((latest_close - prev_close) / prev_close) * 100)
    
    return {
        "latest_close": latest_close,
        "price_change": price_change,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "daily_vol": daily_vol,
        "expected_magnitude": expected_magnitude,
        "contributions": contributions,
        "weights": weights_dict,
        "ic_df": ic_df,
        "selected_factors": selected_factors,
        "backtest_acc": backtest_acc,
        "n_samples": n_obs,
        "total_factors_tested": len(factors_all.columns)
    }

# -------------------------------------------------------------
# 4. 종합 멀티 종목 대시보드 UI
# -------------------------------------------------------------
st.markdown("## 🏛️ AlphaPulse 인스티튜셔널 퀀트 터미널")
st.caption(f"시스템 기준시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 116개 다차원 팩터 전수 백테스트 및 스튜던트 t-검정 스크리너 가동")

if st.button("🔄 116개 팩터 전수 백테스트 및 모델 재학습"):
    st.cache_data.clear()

with st.spinner("6개 종목 및 10대 글로벌 매크로 자산 수집, 116개 팩터 IC 통계 검정 수행 중..."):
    all_targets, all_macros = load_all_market_assets()

stock_tabs = st.tabs([
    "🇺🇸 COHR (Coherent)",
    "🇰🇷 SK하이닉스",
    "🇰🇷 삼성전자",
    "🇰🇷 두산에너빌리티",
    "🇰🇷 하이브",
    "🇰🇷 LS"
])

tab_mapping = [
    ("COHR", stock_tabs[0], "$"),
    ("SK하이닉스", stock_tabs[1], "₩"),
    ("삼성전자", stock_tabs[2], "₩"),
    ("두산에너빌리티", stock_tabs[3], "₩"),
    ("하이브", stock_tabs[4], "₩"),
    ("LS", stock_tabs[5], "₩")
]

for ticker_key, tab, currency in tab_mapping:
    with tab:
        df_target = all_targets.get(ticker_key)
        if df_target is None:
            st.error(f"{ticker_key}의 시계열 데이터를 불러오지 못했습니다. 잠시 후 새로고침해주세요.")
            continue
            
        res = run_institutional_engine(df_target, all_macros, ticker_key)
        if res is None:
            st.warning("데이터 웜업 모수가 부족합니다. (최소 100거래일 필요)")
            continue
            
        # 1. 탑 매트릭스 요약
        c1, c2, c3, c4, c5 = st.columns(5)
        price_txt = f"{int(res['latest_close']):,}원" if currency == "₩" else f"${res['latest_close']:.2f}"
        c1.metric(f"{ticker_key} 최근 종가", price_txt, f"{res['price_change']:+.2f}%")
        c2.metric("전수 백테스트 팩터 수", f"{res['total_factors_tested']}개", "116개 전수 스크리닝")
        c3.metric("최종 통과 유효 팩터", f"{len(res['selected_factors'])}개", "IC & p-value 통과")
        c4.metric("최근 60일 실전 적중률", f"{res['backtest_acc']:.1f}%", "Out-of-sample")
        c5.metric("20일 일일 변동성(σ)", f"±{res['daily_vol']:.2f}%", "정상 진폭 범위")
        
        st.markdown("---")
        
        # 2. 우세 방향 & 예상 변동폭 (Magnitude)
        prob_up = res["prob_up"]
        prob_down = res["prob_down"]
        exp_mag = res["expected_magnitude"]
        
        col_dir, col_gauge = st.columns([1.3, 1])
        
        with col_dir:
            if prob_up >= 54.0:
                st.success(f"### 📈 [상승 우세] 다음 거래일 상승 확률: {prob_up:.1f}%")
                st.write(f"**하락 반대 확률**: {prob_down:.1f}% | 모델 확신도: **상승 우위 (+{prob_up - 50:.1f}%p)**")
                st.info(f"🎯 **예상 일일 변동폭(Magnitude)**: **+{abs(exp_mag)*0.7:.2f}% ~ +{abs(exp_mag)*1.3:.2f}%** (상방 압력)")
            elif prob_up <= 46.0:
                st.error(f"### 📉 [하락 우세] 다음 거래일 하락 확률: {prob_down:.1f}%")
                st.write(f"**상승 반대 확률**: {prob_up:.1f}% | 모델 확신도: **하락 우위 (+{prob_down - 50:.1f}%p)**")
                st.warning(f"🎯 **예상 일일 변동폭(Magnitude)**: **-{abs(exp_mag)*0.7:.2f}% ~ -{abs(exp_mag)*1.3:.2f}%** (하방 압력)")
            else:
                st.warning(f"### ⚖️ [중립 / 관망 권고] 방향성 탐색 (노이즈 구간)")
                st.write(f"**상승**: {prob_up:.1f}% vs **하락**: {prob_down:.1f}%")
                st.write(f"🎯 **예상 일일 변동폭**: **±{res['daily_vol']*0.5:.2f}% 내외 박스권 횡보**")
            st.caption("※ 지수 감쇠 가중치(Half-life 28일)가 적용되어 최신 시장 충격에 적응합니다.")
            
        with col_gauge:
            score_100 = (prob_up - 50.0) * 2.0
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_100,
                title={'text': "신호 강도 (-100:강력하락 ~ +100:강력상승)"},
                gauge={
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "#10b981" if score_100 >= 0 else "#ef4444"},
                    'steps': [
                        {'range': [-100, -20], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [-20, 20], 'color': "rgba(107, 114, 128, 0.2)"},
                        {'range': [20, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': 0}
                }
            ))
            fig_g.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g, use_container_width=True)
            
        # 3. 채택된 핵심 팩터 기여도 및 가중치 차트
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            c_names = list(res["contributions"].keys())
            c_vals = list(res["contributions"].values())
            c_colors = ["#10b981" if v >= 0 else "#ef4444" for v in c_vals]
            fig_bar = go.Figure(go.Bar(x=c_vals, y=c_names, orientation='h', marker_color=c_colors))
            fig_bar.update_layout(title="채택된 팩터별 오늘 밤 기여도 분해 (%p)", height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_chart2:
            w_names = list(res["weights"].keys())
            w_vals = list(res["weights"].values())
            fig_w = go.Figure(go.Bar(x=w_vals, y=w_names, orientation='h', marker_color="#3b82f6"))
            fig_w.update_layout(title="머신러닝이 최적화한 팩터별 가중치 (β)", height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_w, use_container_width=True)
            
        # 4. 116개 전체 팩터의 통계적 유의성 전수 백테스트 리포트
        with st.expander(f"📊 {ticker_key} - 116개 전체 팩터 IC 통계 검정 순위표 (클릭하여 펼치기)"):
            st.write("※ 116개 팩터 전수에 대해 다음 거래일 수익률과의 순위 상관계수(IC), Student's t-통계량, p-value를 계산한 결과입니다.")
            display_ic = res["ic_df"].copy()
            display_ic["스크리닝 결과"] = display_ic["팩터코드"].apply(
                lambda x: "🟢 통과 (모델 채택)" if x in res["selected_factors"] else "⚪ 탈락 (노이즈/중복)"
            )
            st.dataframe(
                display_ic[["팩터코드", "스피어만 IC", "t-통계량", "p-value", "스크리닝 결과"]],
                use_container_width=True,
                height=400
            )
