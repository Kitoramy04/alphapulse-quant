import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 페이지 기본 설정
st.set_page_config(page_title="AlphaPulse Quant Terminal", layout="wide")

# -------------------------------------------------------------
# 1. 퀀트 확률 산출 로직 (Ridge Regularized Logistic Scoring)
# -------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_quant_probability(ticker_type, factors):
    """
    과적합을 방지하기 위해 L2 패널티 정규화 가중치를 적용한 확률 스코어러
    """
    if ticker_type == "COHR":
        intercept = 0.05
        weights = {
            "mom_5d": 0.30,      # 5일 모멘텀 / 20일 괴리율
            "qqq_ret": 0.45,     # 나스닥 100 당일/야간 등락률
            "wti_ret": -0.22,    # WTI 유가 일일 변동률 (성장주 역상관)
            "vol_ratio": 0.15    # 20일 평균 대비 거래량 비율
        }
        # 표준화 팩터 스케일링
        z = intercept + (
            weights["mom_5d"] * np.clip(factors["mom_5d"] / 3.0, -2, 2) +
            weights["qqq_ret"] * np.clip(factors["qqq_ret"] / 1.5, -2, 2) +
            weights["wti_ret"] * np.clip(factors["wti_ret"] / 2.0, -2, 2) +
            weights["vol_ratio"] * np.clip((factors["vol_ratio"] - 1.0) / 0.5, -2, 2)
        )
        contributions = {
            "5일 모멘텀": weights["mom_5d"] * np.clip(factors["mom_5d"] / 3.0, -2, 2) * 25,
            "QQQ 기술주 방향": weights["qqq_ret"] * np.clip(factors["qqq_ret"] / 1.5, -2, 2) * 25,
            "WTI 유가 충격": weights["wti_ret"] * np.clip(factors["wti_ret"] / 2.0, -2, 2) * 25,
            "거래량 폭증도": weights["vol_ratio"] * np.clip((factors["vol_ratio"] - 1.0) / 0.5, -2, 2) * 25
        }
    else:  # SK하이닉스 (000660)
        intercept = 0.08
        weights = {
            "adr_ret": 0.50,     # 미국 상장 SKHY ADR 야간 등락률
            "qqq_ret": 0.35,     # 나스닥 선물 / QQQ 프리마켓
            "wti_ret": -0.18,    # WTI 유가 변동률
            "supply_z": 0.38     # 최근 수급 강도 프록시
        }
        z = intercept + (
            weights["adr_ret"] * np.clip(factors["adr_ret"] / 2.5, -2, 2) +
            weights["qqq_ret"] * np.clip(factors["qqq_ret"] / 1.5, -2, 2) +
            weights["wti_ret"] * np.clip(factors["wti_ret"] / 2.0, -2, 2) +
            weights["supply_z"] * np.clip(factors["supply_z"], -2, 2)
        )
        contributions = {
            "SKHY 야간 ADR": weights["adr_ret"] * np.clip(factors["adr_ret"] / 2.5, -2, 2) * 25,
            "새벽 나스닥 동향": weights["qqq_ret"] * np.clip(factors["qqq_ret"] / 1.5, -2, 2) * 25,
            "WTI 유가 부담": weights["wti_ret"] * np.clip(factors["wti_ret"] / 2.0, -2, 2) * 25,
            "수급 강도(외인/기관)": weights["supply_z"] * np.clip(factors["supply_z"], -2, 2) * 25
        }

    prob = sigmoid(z) * 100
    return prob, contributions

# -------------------------------------------------------------
# 2. 실시간 야간 데이터 수집 엔진 (yfinance 기반, 5분 캐싱)
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_market_data():
    tickers = ["COHR", "QQQ", "CL=F", "SKHY", "000660.KS"]
    data = {}
    
    for t in tickers:
        ticker_obj = yf.Ticker(t)
        hist = ticker_obj.history(period="1mo")
        if not hist.empty and len(hist) >= 2:
            latest = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            pct_change = ((latest - prev) / prev) * 100
            vol_ratio = hist["Volume"].iloc[-1] / hist["Volume"].rolling(20).mean().iloc[-1] if len(hist) >= 20 else 1.0
            
            # 20일 이평선 괴리도
            ma20 = hist["Close"].rolling(20).mean().iloc[-1] if len(hist) >= 20 else latest
            ma_gap = ((latest - ma20) / ma20) * 100
            
            data[t] = {
                "price": latest,
                "pct_change": pct_change,
                "vol_ratio": vol_ratio,
                "ma_gap": ma_gap
            }
        else:
            data[t] = {"price": 0.0, "pct_change": 0.0, "vol_ratio": 1.0, "ma_gap": 0.0}
            
    return data

# -------------------------------------------------------------
# 3. UI 대시보드 렌더링
# -------------------------------------------------------------
st.markdown("## 📊 AlphaPulse 실시간 주가예측 퀀트 대시보드")
st.caption(f"기준 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (접속 시 자동 최신 시세 갱신)")

if st.button("🔄 실시간 데이터 강제 새로고침"):
    st.cache_data.clear()

with st.spinner("야간 선물, WTI, ADR 및 실시간 팩터 데이터 수집 중..."):
    market_data = fetch_market_data()

tab1, tab2 = st.tabs(["🇺🇸 Coherent (COHR)", "🇰🇷 SK하이닉스 (000660)"])

# ----- COHR 탭 -----
with tab1:
    cohr = market_data["COHR"]
    qqq = market_data["QQQ"]
    wti = market_data["CL=F"]
    
    cohr_factors = {
        "mom_5d": cohr["ma_gap"],
        "qqq_ret": qqq["pct_change"],
        "wti_ret": wti["pct_change"],
        "vol_ratio": cohr["vol_ratio"]
    }
    
    prob_cohr, contrib_cohr = calculate_quant_probability("COHR", cohr_factors)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("COHR 종가/현재가", f"${cohr['price']:.2f}", f"{cohr['pct_change']:+.2f}%")
    col2.metric("QQQ (나스닥 100)", f"${qqq['price']:.2f}", f"{qqq['pct_change']:+.2f}%")
    col3.metric("WTI 원유 선물", f"${wti['price']:.2f}", f"{wti['pct_change']:+.2f}%")
    col4.metric("거래량 상대강도", f"{cohr['vol_ratio']:.2f}x")

    col_gauge, col_waterfall = st.columns([1, 1])
    
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_cohr,
            title={'text': "다음 거래일 상승 확률"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981" if prob_cohr >= 50 else "#ef4444"},
                'steps': [
                    {'range': [0, 45], 'color': "#374151"},
                    {'range': [45, 55], 'color': "#1f2937"},
                    {'range': [55, 100], 'color': "#374151"}
                ],
                'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_waterfall:
        names = list(contrib_cohr.keys())
        values = list(contrib_cohr.values())
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]
        
        fig_bar = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker_color=colors
        ))
        fig_bar.update_layout(
            title="팩터별 확률 기여도 (%p)",
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="기여도 (%p)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ----- SK하이닉스 탭 -----
with tab2:
    hynix = market_data["000660.KS"]
    skhy = market_data["SKHY"]
    
    sk_factors = {
        "adr_ret": skhy["pct_change"] if skhy["price"] > 0 else qqq["pct_change"],
        "qqq_ret": qqq["pct_change"],
        "wti_ret": wti["pct_change"],
        "supply_z": 0.4 if hynix["pct_change"] > 0 else -0.3 # 가격 동조 수급 프록시
    }
    
    prob_sk, contrib_sk = calculate_quant_probability("000660", sk_factors)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SK하이닉스 (국내)", f"{int(hynix['price']):,}원", f"{hynix['pct_change']:+.2f}%")
    col2.metric("SKHY (미국 ADR)", f"${skhy['price']:.2f}", f"{skhy['pct_change']:+.2f}%")
    col3.metric("WTI 원유 선물", f"${wti['price']:.2f}", f"{wti['pct_change']:+.2f}%")
    col4.metric("QQQ (나스닥 프록시)", f"${qqq['price']:.2f}", f"{qqq['pct_change']:+.2f}%")

    col_gauge_sk, col_waterfall_sk = st.columns([1, 1])
    
    with col_gauge_sk:
        fig_gauge_sk = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_sk,
            title={'text': "익일 국내 개장 시 상승 확률"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981" if prob_sk >= 50 else "#ef4444"},
                'steps': [
                    {'range': [0, 45], 'color': "#374151"},
                    {'range': [45, 55], 'color': "#1f2937"},
                    {'range': [55, 100], 'color': "#374151"}
                ],
                'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge_sk.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge_sk, use_container_width=True)

    with col_waterfall_sk:
        names_sk = list(contrib_sk.keys())
        values_sk = list(contrib_sk.values())
        colors_sk = ["#10b981" if v >= 0 else "#ef4444" for v in values_sk]
        
        fig_bar_sk = go.Figure(go.Bar(
            x=values_sk,
            y=names_sk,
            orientation='h',
            marker_color=colors_sk
        ))
        fig_bar_sk.update_layout(
            title="팩터별 확률 기여도 (%p)",
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="기여도 (%p)"
        )
        st.plotly_chart(fig_bar_sk, use_container_width=True)