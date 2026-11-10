
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
    page_title="월별 기온 분석 및 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 기온 분석 및 미래 예측")
st.write("1907년부터 2025년까지의 실제 기온을 분석하고 미래 기온을 예측합니다.")

st.divider()

# ----------------------------------
# 1. 데이터 불러오기
# ----------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "temperature.csv",
        encoding="cp949"
    )

    # 날짜에서 월과 연도 추출
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

    # 두 자리 연도를 실제 연도로 변환
    yy = pd.to_numeric(date_info[1], errors="coerce")

    # 1999년에서 2000년으로 넘어가는 시점 자동 확인
    century_change = yy.diff().lt(0).cumsum()

    df["연도"] = 1900 + yy + century_change * 100

    # 평균기온 숫자로 변환
    df["평균기온(℃)"] = pd.to_numeric(
        df["평균기온(℃)"],
        errors="coerce"
    )

    # 실제 데이터 1907~2025년만 사용
    df = df[
        (df["연도"] >= 1907) &
        (df["연도"] <= 2025)
    ].copy()

    df = df.dropna(subset=["월"])

    return df


# ----------------------------------
# 2. 데이터 분석
# ----------------------------------

try:

    df = load_data()

    # 월 선택
    month = st.selectbox(
        "📅 분석할 월을 선택하세요.",
        list(range(1, 13)),
        format_func=lambda x: f"{x}월"
    )

    # 미래 예측 연도
    forecast_year = st.number_input(
        "🔮 예측할 미래 연도를 입력하세요.",
        min_value=2026,
        max_value=2100,
        value=2030,
        step=1
    )

    # 선택한 월의 평균기온
    selected_df = df[df["월"] == month]

    actual = selected_df.groupby("연도")[
        "평균기온(℃)"
    ].mean()

    # 1907~2025년 전체 연도 구성
    years = np.arange(1907, 2026)

    actual = actual.reindex(years)

    # 내부 결측치만 보간
    training_data = actual.interpolate(
        method="linear",
        limit_area="inside"
    ).dropna()

    if len(training_data) < 10:

        st.warning("예측에 필요한 데이터가 부족합니다.")

    else:

        # 학습 데이터 준비
        X = training_data.index.to_numpy().reshape(-1, 1)
        y = training_data.to_numpy(dtype=float)

        # 미래 연도
        future_years = np.arange(
            2026,
            int(forecast_year) + 1
        )

        future_X = future_years.reshape(-1, 1)

        # ----------------------------------
        # 3. 알고리즘 1 : 선형 회귀
        # ----------------------------------

        linear_model = LinearRegression()

        linear_model.fit(X, y)

        linear_history = linear_model.predict(X)

        linear_future = linear_model.predict(future_X)

        # ----------------------------------
        # 4. 알고리즘 2 : 랜덤 포레스트
        # ----------------------------------

        forest_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )

        forest_model.fit(X, y)

        forest_history = forest_model.predict(X)

        forest_future = forest_model.predict(future_X)

        # ----------------------------------
        # 5. 알고리즘 3 : Holt 지수평활법
        # ----------------------------------

        # 중요: Pandas 인덱스 대신 NumPy 배열 사용
        train_values = y.astype(float)

        holt_model = Holt(
            train_values,
            damped_trend=True,
            initialization_method="estimated"
        )

        holt_fit = holt_model.fit(
            optimized=True
        )

        holt_history = np.asarray(
            holt_fit.fittedvalues
        )

        holt_future = np.asarray(
            holt_fit.forecast(
                steps=len(future_years)
            )
        )

        # ----------------------------------
        # 6. 그래프 생성
        # ----------------------------------

        fig = go.Figure()

        # 실제 평균기온
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="실제 평균기온",
            line=dict(color="black", width=2),
            connectgaps=False
        ))

        # 선형 회귀 학습 결과
        fig.add_trace(go.Scatter(
            x=training_data.index,
            y=linear_history,
            mode="lines",
            name="선형 회귀 학습",
            line=dict(color="red", dash="dash", width=2)
        ))

        # 선형 회귀 미래 예측
        fig.add_trace(go.Scatter(
            x=np.r_[2025, future_years],
            y=np.r_[
                linear_model.predict([[2025]])[0],
                linear_future
            ],
            mode="lines+markers",
            name="선형 회귀 예측",
            line=dict(color="red", width=3)
        ))

        # 랜덤 포레스트 학습 결과
        fig.add_trace(go.Scatter(
            x=training_data.index,
            y=forest_history,
            mode="lines",
            name="랜덤 포레스트 학습",
            line=dict(color="blue", dash="dash", width=2)
        ))

        # 랜덤 포레스트 미래 예측
        fig.add_trace(go.Scatter(
            x=np.r_[2025, future_years],
            y=np.r_[
                forest_model.predict([[2025]])[0],
                forest_future
            ],
            mode="lines+markers",
            name="랜덤 포레스트 예측",
            line=dict(color="blue", width=3)
        ))

        # Holt 학습 결과
        fig.add_trace(go.Scatter(
            x=training_data.index,
            y=holt_history,
            mode="lines",
            name="Holt 학습",
            line=dict(color="green", dash="dash", width=2)
        ))

        # Holt 미래 예측
        fig.add_trace(go.Scatter(
            x=np.r_[2025, future_years],
            y=np.r_[
                holt_history[-1],
                holt_future
            ],
            mode="lines+markers",
            name="Holt 예측",
            line=dict(color="green", width=3)
        ))

        # 미래 예측 시작점
        fig.add_vline(
            x=2025.5,
            line_dash="dot",
            line_color="gray",
            annotation_text="2026년부터 미래 예측"
        )

        # 미래 영역 표시
        fig.add_vrect(
            x0=2025.5,
            x1=int(forecast_year) + 0.5,
            fillcolor="gray",
            opacity=0.08,
            line_width=0
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
                title="알고리즘 범례",
                orientation="h",
                y=1.15,
                x=0
            )
        )

        fig.update_xaxes(
            range=[1907, int(forecast_year) + 1],
            dtick=10
        )

        # 그래프 출력
        st.subheader("📈 실제 기온과 미래 예측 비교")

        st.caption(
            "검은색: 실제 데이터 | "
            "점선: 과거 학습 결과 | "
            "실선: 미래 예측"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------
        # 7. 알고리즘 설명
        # ----------------------------------

        st.divider()

        st.subheader("🧠 예측 알고리즘 설명")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔴 선형 회귀")
            st.write(
                "과거 기온의 전체적인 증가와 감소 추세를 "
                "직선으로 학습합니다. "
                "이 추세를 미래까지 연장하여 예측합니다."
            )
            st.caption("그래프 특징: 직선 형태")

        with col2:
            st.markdown("### 🔵 랜덤 포레스트")
            st.write(
                "여러 결정 트리가 과거 기온 데이터를 "
                "학습하고 각각의 예측 결과를 평균 내어 "
                "미래 기온을 계산합니다."
            )
            st.caption("그래프 특징: 구간별 패턴")

        with col3:
            st.markdown("### 🟢 Holt")
            st.write(
                "최근 기온과 변화 추세에 가중치를 주어 "
                "미래 기온을 부드럽게 예측합니다."
            )
            st.caption("그래프 특징: 추세를 반영한 곡선")

        # ----------------------------------
        # 8. 예측 결과 표
        # ----------------------------------

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

        st.success(
            f"{month}월의 {forecast_year}년까지 예측이 완료되었습니다."
        )

        st.info(
            "예측 결과는 과거 평균기온만 학습한 통계적 추정치입니다. "
            "실제 기온과 다를 수 있으며 온실가스, 기후변화 등 "
            "외부 요인은 반영하지 않았습니다."
        )

except FileNotFoundError:
    st.error(
        "temperature.csv 파일이 없습니다. "
        "app.py와 같은 폴더에 업로드해 주세요."
    )

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
