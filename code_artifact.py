import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────
# 1. 설정 및 자산 티커 매핑 (달러화 화폐 통일)
# ─────────────────────────────────────────────────────────────────────────
# 공정한 비교를 위해 코스피를 미국 상장 달러 표시 ETF인 EWY로 설정
ASSETS = {
    '코스피(EWY)': 'EWY',  
    'S&P500(SPY)': 'SPY',
    '금(GLD)': 'GLD',
    '중기채(IEF)': 'IEF'
}

OFFENSIVE = ['코스피(EWY)', 'S&P500(SPY)', '금(GLD)']

# ─────────────────────────────────────────────────────────────────────────
# 2. 데이터 다운로드 및 3개월 수익률 계산
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_data_and_calc_momentum(start_year, end_year):
    tickers = list(ASSETS.values())
    
    start_dt = f"{start_year - 1}-09-01"
    end_dt = f"{end_year}-12-31"
    
    raw = yf.download(tickers, start=start_dt, end=end_dt, auto_adjust=True, progress=False)['Close']
    monthly = raw.resample('ME').last()
    
    # 컬럼 이름을 한글 이름으로 변경
    inv_map = {v: k for k, v in ASSETS.items()}
    monthly.rename(columns=inv_map, inplace=True)
    
    # 3개월 수익률 계산 (모든 자산이 달러 기준으로 공정하게 계산됨)
    mom_3m = monthly / monthly.shift(3) - 1
    
    return monthly, mom_3m

# ─────────────────────────────────────────────────────────────────────────
# 3. 백테스트 엔진
# ─────────────────────────────────────────────────────────────────────────
def run_custom_momentum(monthly, mom_3m, start_year):
    sim_dates = [d for d in monthly.index if d.year >= start_year]
    records = []
    capital = 1.0
    
    for i in range(len(sim_dates) - 1):
        date = sim_dates[i]
        next_date = sim_dates[i+1]
        
        # 현재 날짜에 모든 공격 자산의 데이터가 존재하는지 확인 (상장일 이후 자동 시작용)
        if monthly.loc[date, OFFENSIVE].isna().any() or mom_3m.loc[date, OFFENSIVE].isna().any():
            continue 
            
        off_moms = mom_3m.loc[date, OFFENSIVE]
        
        # 💡 투자 로직 판단
        if (off_moms < 0).all():
            # [방어 모드] 3개 공격 자산 모두 마이너스일 때
            ief_mom = mom_3m.loc[date, '중기채(IEF)']
            
            # 미국 중기채(IEF) 3개월 수익률이 플러스(+)면 IEF 매수, 마이너스(-)면 현금 보유
            if pd.notna(ief_mom) and ief_mom > 0:
                chosen = '중기채(IEF)'
                mode = "🛡️ 방어(채권)"
            else:
                chosen = '현금'
                mode = "🛡️ 방어(현금)"
        else:
            # [공격 모드] 셋 중 하나라도 플러스면, 3개월 수익률 1등에 올인
            mode = "⚔️ 공격"
            chosen = off_moms.idxmax()
            
        # 수익률 계산 (현금일 때는 자산 변동 없음)
        if chosen == '현금':
            ret = 0.0
        else:
            if pd.notna(monthly.loc[next_date, chosen]) and monthly.loc[date, chosen] > 0:
                ret = monthly.loc[next_date, chosen] / monthly.loc[date, chosen] - 1
            else:
                ret = 0.0
            
        capital *= (1 + ret)
        
        # 출력용 모멘텀 값 처리
        if chosen == '현금':
            display_mom = "0.00%"
        else:
            display_mom = f"{mom_3m.loc[date, chosen]*100:.2f}%"
        
        records.append({
            'date': next_date,
            '투자 모드': mode,
            '매수 종목': chosen,
            '3M 모멘텀': display_mom,
            '월 수익률': ret,
            '누적 자산': capital
        })
        
    return pd.DataFrame(records).set_index('date')

# ─────────────────────────────────────────────────────────────────────────
# 4. 스트림릿 웹앱 UI
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="달러 통일 맞춤형 듀얼 모멘텀", page_icon="📈", layout="wide")
st.title("📈 달러 통일 맞춤형 듀얼 모멘텀 백테스터")
st.markdown("""
**[전략 룰 - 화폐 가치 달러화 통일]**
1. 매월 말 **코스피(EWY), S&P500(SPY), 금(GLD)**의 최근 3개월 수익률을 확인합니다. (모두 달러 기준)
2. 셋 중 **하나라도 0% 이상**이면, 3개월 수익률이 **가장 높은 종목에 100% 올인**합니다.
3. 셋 다 **0% 미만(마이너스)**이면, **미국 중기채(IEF)의 3개월 수익률을 확인**합니다.
   - IEF 3개월 수익률이 **0% 이상이면 IEF 보유**
   - IEF 3개월 수익률마저 **0% 미만이면 전량 현금(수익률 0%) 보유**
""")

with st.sidebar:
    st.header("⚙️ 백테스트 설정")
    s_year = st.number_input("시작 연도 (ETF 상장일 이후 자동 시작)", min_value=2000, max_value=2023, value=2000)
    e_year = st.number_input("종료 연도", min_value=2010, max_value=2026, value=2024)
    run_btn = st.button("🚀 백테스트 실행", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("데이터 수집 중..."):
        monthly_df, mom_3m_df = download_data_and_calc_momentum(s_year, e_year)
        
    with st.spinner("시뮬레이션 진행 중..."):
        res_df = run_custom_momentum(monthly_df, mom_3m_df, s_year)
        
    if res_df.empty:
        st.error("데이터가 부족합니다. 시작 연도를 조절해 보세요.")
    else:
        # 성과 지표 계산 (Numpy 날짜 에러 패치 적용 버전)
        years = len(res_df) / 12  
        if years <= 0: years = 1
        cagr = (res_df['누적 자산'].iloc[-1] ** (1/years) - 1) * 100
        mdd = ((res_df['누적 자산'] - res_df['누적 자산'].cummax()) / res_df['누적 자산'].cummax()).min() * 100
        sharpe = (res_df['월 수익률'].mean() / res_df['월 수익률'].std()) * np.sqrt(12)
        win_rate = (res_df['월 수익률'] > 0).mean() * 100
        
        spy_prices = monthly_df.loc[res_df.index, 'S&P500(SPY)']
        spy_rets = spy_prices.pct_change().dropna()
        spy_cum = (1 + spy_rets).cumprod()
        spy_years = len(spy_cum) / 12
        if spy_years <= 0: spy_years = 1
        spy_cagr = (spy_cum.iloc[-1] ** (1/spy_years) - 1) * 100 if not spy_cum.empty else 0
        spy_mdd = ((spy_cum - spy_cum.cummax()) / spy_cum.cummax()).min() * 100 if not spy_cum.empty else 0

        st.subheader("📊 백테스트 성과 요약")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("연평균 수익률 (CAGR)", f"{cagr:.2f}%", f"SPY 대비 {cagr - spy_cagr:+.2f}%")
        c2.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%", f"SPY 대비 {mdd - spy_mdd:+.2f}%", delta_color="inverse")
        c3.metric("샤프 지수", f"{sharpe:.2f}")
        c4.metric("월간 승률", f"{win_rate:.1f}%")

        st.subheader("📈 누적 수익률 비교 (Log Scale)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(res_df.index, res_df['누적 자산'], label=f"달러 통일 전략 (CAGR: {cagr:.1f}%)", color='#27ae60', linewidth=2)
        ax.plot(spy_cum.index, spy_cum, label=f"S&P 500 B&H (CAGR: {spy_cagr:.1f}%)", color='#3498db', linestyle='--', linewidth=1.5)
        ax.set_yscale('log')
        ax.set_ylabel("Growth of $1")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.subheader("📋 월별 리밸런싱 및 투자 내역")
        display_df = res_df.copy()
        display_df['월 수익률'] = (display_df['월 수익률'] * 100).round(2).astype(str) + '%'
        display_df['누적 자산'] = "$" + display_df['누적 자산'].round(3).astype(str)
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)
