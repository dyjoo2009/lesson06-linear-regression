import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_annual_mean() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = df.columns.str.strip()

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= LAST_YEAR]

    # 연도별 평균기온과 관측일 수(평균기온이 기록된 날)
    annual = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일="count")
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    annual = annual[annual["관측일"] >= MIN_DAYS].reset_index(drop=True)
    return annual


st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온으로 만든 회귀 직선으로 원하는 해의 기온을 예상해 봅니다.")

try:
    annual = load_annual_mean()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(annual) < 2:
    st.warning("조건에 맞는 연도가 부족해서 회귀 직선을 만들 수 없어요.")
    st.stop()

x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
n_years = len(annual)

# 최근 20년만으로 구한 직선
RECENT_YEARS = 20
recent = annual[annual["연도"] > end_year - RECENT_YEARS]
recent_ok = len(recent) >= 3
if recent_ok:
    r_slope, r_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    r_corr = float(np.corrcoef(recent["연도"], recent["연평균기온"])[0, 1])

# ---- 사용한 데이터 정보 ----
st.subheader("직선을 만든 데이터")
c1, c2, c3 = st.columns(3)
c1.metric("사용한 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
st.caption(
    f"{LAST_YEAR}년 이후 데이터와 관측일이 {MIN_DAYS}일 미만인 해는 제외했어요."
)

# ---- 연도 선택 & 예측 ----
st.subheader("예상 기온")
year = st.slider("연도를 골라 보세요", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년의 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if year < start_year or year > end_year:
    st.info(
        f"{year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이라서, "
        "직선을 그대로 늘려 계산한 값이에요. 참고용으로만 봐 주세요."
    )

# ---- 100년당 기온 변화 ----
st.subheader("100년에 몇 도 오를까?")
st.markdown(
    f"""
    <div style="text-align:center; padding:0.8rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">전체 기간({start_year}~{end_year}년) 기준</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{slope * 100:+.2f} °C / 100년</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 전체 기간 vs 최근 20년 ----
st.subheader("전체 기간 vs 최근 20년")
col_all, col_recent = st.columns(2)


def slope_card(col, title, period, value, n, corr):
    col.markdown(
        f"""
        <div style="text-align:center; padding:0.8rem; border:1px solid rgba(128,128,128,0.4);
                    border-radius:12px;">
            <div style="font-size:1.1rem; font-weight:700;">{title}</div>
            <div style="font-size:0.9rem; opacity:0.7;">{period} · {n}개 해</div>
            <div style="font-size:2.4rem; font-weight:800; line-height:1.3;">{value * 100:+.2f} °C</div>
            <div style="font-size:0.9rem; opacity:0.7;">100년당 · 상관계수 {corr:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


slope_card(col_all, "전체 기간", f"{start_year}~{end_year}년", slope, n_years, r)
if recent_ok:
    slope_card(
        col_recent,
        f"최근 {RECENT_YEARS}년",
        f"{int(recent['연도'].min())}~{end_year}년",
        r_slope,
        len(recent),
        r_corr,
    )
    ratio = r_slope / slope if slope != 0 else float("nan")
    if np.isfinite(ratio):
        st.info(
            f"최근 {RECENT_YEARS}년의 상승 속도는 전체 기간의 약 {ratio:.1f}배예요. "
            "다만 20년은 짧아서 해마다의 들쭉날쭉함에 크게 흔들릴 수 있어요."
        )
else:
    col_recent.warning(f"최근 {RECENT_YEARS}년 안에 데이터가 3개 해 미만이라 계산할 수 없어요.")

# ---- 상관계수 ----
st.subheader("산점도와 회귀 직선")
st.metric("상관계수 (r, 전체 기간)", f"{r:.3f}")
st.caption(f"회귀식(전체 기간): 기온 = {slope:.4f} × 연도 + ({intercept:.2f})")

# ---- 그래프 ----
line_x0 = min(start_year, year)
line_x1 = max(end_year, year)
line_x = np.array([line_x0, line_x1], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
        hoverinfo="skip",
    )
)
if recent_ok:
    rx = np.array([int(recent["연도"].min()), end_year], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=r_slope * rx + r_intercept,
            mode="lines",
            name=f"최근 {RECENT_YEARS}년 직선",
            line=dict(width=3, color="royalblue", dash="dash"),
            hoverinfo="skip",
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", color="gold", line=dict(width=1.5, color="black")),
        hovertemplate=f"{year}년 예상<br>{pred:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 표 보기"):
    show = annual.rename(columns={"관측일": "관측일 수"}).copy()
    show["연평균기온"] = show["연평균기온"].round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)
