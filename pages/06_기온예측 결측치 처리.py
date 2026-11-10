
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import Holt
import warnings

warnings.filterwarnings("ignore")

# 페이지 설정
st.set_page_config(
    page_title="기온 예측 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 기온 분석 및 미래 예측")
st.write("1907년부터 2025년까지의 실제 기온 데이터를 분석하고 미래 기온을 예측합니다.")

st.divider()

# --------------------------------
# 1. 데이터 불러오기
# --------------------------------

@st.cache_data
def load_data():

    try:
        df = pd.read_csv("temperature.csv", encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv("temperature.csv", encoding="utf-8-sig")

    # 날짜에서 월과 두 자리 연도 추출
    date_info = df["일시"].astype(str).str.extract(
        r"([A-Za-z]{3})-(\d{2})"
    )

    month_dict = {
        "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,
        "Oct": 10, "Nov": 11, "Dec": 12
    }

    df["월"] = date_info[0].map(month_dict)

    # 연도 변환
    yy = pd.to_numeric(date_info[1], errors="coerce")

    # 99년에서 00년으로 넘어가는 지점 확인
    century_change = (yy.diff() < 0).fillna(False).cumsum()

    df["연도"] = 1900 + yy + century_change * 100

    # 평균기온 숫자로 변환
    df["평균기온(℃)"] = pd.to_numeric(
        df["평균기온(℃)"], errors="coerce"
    )

    # 실제 데이터만 사용
    df = df[
        (df["연도"] >= 1907) &
        (df["연도"] <= 2025)
    ].copy()

    df = df.dropna(subset=["월"])

    return df


# --------------------------------
# 2. 데이터 실행
# --------------------------------

try:

    df = load_data()

    # 월 선택
    month = st.selectbox(
        "📅 분석할 월을 선택하세요.",
        range(1, 13),
        format_func=lambda x: f"{x}월"
    )

    # 예측 연도 선택
    forecast_year = st.number_input(
        "🔮 예측할 미래 연도를 입력하세요.",
        min_value=2026,
        max_value=2100,
        value=2030,
        step=1
    )

    # 선택한 월의 데이터
    selected_df = df[df["월"] == month]

    # 연도별 평균기온 계산
    actual = selected_df.groupby("연도")["평균기온(℃)"].mean()

    # 1907~2025년 범위
    years = np.arange(1907, 2026)

    # 데이터 정리
    actual = actual.reindex(years)

    # 모델 학습용 결측치 보간
    training_data = actual.interpolate(
        method="linear",
        limit_area="inside"
    ).dropna()

    if len(training_data) < 10:
        st.warning("예측에 필요한 데이터가 부족합니다.")

    else:

        # 학습 데이터
        X = training_data.index.to_numpy().reshape(-1, 1)
        y = training_data.to_numpy()

        future_years = np.arange(
            2026, int(forecast_year) + 1
        )

        future_X = future_years.reshape(-1, 1)

        # --------------------------------
        # 3. 알고리즘 학습
        # --------------------------------

        # ① 선형 회귀
        linear_model = LinearRegression()
        linear_model.fit(X, y)

        linear_history = linear_model.predict(X)
        linear_future = linear_model.predict(future_X)

        # ② 랜덤 포레스트
        forest_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )

        forest_model.fit(X, y)

        forest_history = forest_model.predict(X)
        forest_future = forest_model.predict(future_X)

        # ③ Holt 지수평활법
        holt_model = Holt(
            training_data,
            damped_trend=True,
            initialization_method="estimated"
        )

        holt_fit = holt_model.fit(optimized=True)

        holt_history = np.asarray(holt_fit.fittedvalues)
        holt_future = np.asarray(
            holt_fit.forecast(len(future_years))
        )

        # --------------------------------
        # 4. 그래프 생성
        # --------------------------------

        fig = go.Figure()

        # 실제 평균기온
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="실제 평균기온",
            line=dict(color="black", width=2)
        ))

        # 선형 회귀 과거 학습 결과
        fig.add_trace(go.Scatter(
            x=training_data.index,
            y=linear_history,
            mode="lines",
            name="선형 회귀 (직선 추세)",
            line=dict(color="red", width=2, dash="dash")
        ))

        # 선형 회귀 미래 예측
        fig.add_trace(go.Scatter(
            x=np.r_[2025, future_years],
            y=np.r_[linear_model.predict([[2025]])[0], linear_future],
            mode="lines+markers",
            name="선형 회귀 미래 예측",
            line=dict(color="red", width=3)
        ))

        # 랜덤 포레스트 과거 학습 결과
        fig.add_trace(go.Scatter(
            x=training_data.index,
            y=forest_history,
            mode="lines",
            name="랜덤 포레스트 (구간별 학습)",
            line=dict(color="blue", width=2, dash="dash")
        ))

        # 랜덤 포레스트 미래 예측
        fig.add_trace(go.Scatter(
            x=np.r_[2025, future_years],
            y=np.r_[forest_model.predict([[2025]])[0], forest_future],
            mode="lines+markers",
            name="랜덤 포레스트 미래 예측",
            line=dict(color="blue", width=3)
        ))

        # Holt 과거 학습 결과
        fig.add_trace(go.Scatter(
            x=training_data.index,
            y=holt_history,
            mode="lines",
            name="Holt (최근 추세 반영)",
            line=dict(color="green", width=2, dash="dash")
        ))

        # Holt 미래 예측
        fig.add_trace(go.Scatter(
            x=np.r_[2025, future_years],
            y=np.r_[holt_history[-1], holt_future],
            mode="lines+markers",
            name="Holt 미래 예측",
            line=dict(color="green", width=3)
        ))

        # 예측 시작 지점 표시
        fig.add_vline(
            x=2025,
            line_dash="dot",
            line_color="gray",
            annotation_text="예측 시작 (2026년)"
        )

        # 그래프 디자인
        fig.update_layout(
            title=f"{month}월 평균기온 분석 및 미래 예측",
            xaxis_title="연도",
            yaxis_title="평균기온 (℃)",
            template="plotly_white",
            hovermode="x unified",
            height=650,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )

        fig.update_xaxes(
            range=[1907, int(forecast_year)],
            dtick=10
        )

        st.subheader("📈 실제 기온과 미래 예측 비교")
        st.caption("검은색: 실제 기온 | 점선: 과거 학습 결과 | 실선: 미래 예측")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------
        # 5. 알고리즘 설명
        # --------------------------------

        st.divider()
        st.subheader("🧠 예측 알고리즘 알아보기")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔴 선형 회귀")
            st.write(
                "과거 기온의 전체적인 변화 방향을 직선으로 나타내고 "
                "그 추세가 이어진다고 가정하여 미래를 예측합니다."
            )
            st.caption("특징: 직선 형태의 추세 예측")

        with col2:
            st.markdown("### 🔵 랜덤 포레스트")
            st.write(
                "여러 결정 트리가 과거 기온을 구간별로 학습하고 "
                "각 트리의 예측 결과를 평균 내어 값을 계산합니다."
            )
            st.caption("특징: 구간별 패턴 학습")

        with col3:
            st.markdown("### 🟢 Holt")
            st.write(
                "최근 기온과 변화 추세를 반영하여 "
                "미래의 기온을 부드러운 곡선으로 예측합니다."
            )
            st.caption("특징: 최근 변화와 추세 반영")

        # --------------------------------
        # 6. 미래 예측 결과 표
        # --------------------------------

        st.divider()
        st.subheader("🔮 미래 기온 예측 결과")

        prediction = pd.DataFrame({
            "연도": future_years,
            "선형 회귀(℃)": linear_future,
            "랜덤 포레스트(℃)": forest_future,
            "Holt(℃)": holt_future
        })

        st.dataframe(
            prediction.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "예측값은 과거 평균기온만을 학습한 통계적 추정치입니다. "
            "실제 미래 기온과 다를 수 있으며, 기후변화나 온실가스 배출량 등 "
            "외부 요인은 반영하지 않았습니다."
        )

except FileNotFoundError:
    st.error("temperature.csv 파일을 찾을 수 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
