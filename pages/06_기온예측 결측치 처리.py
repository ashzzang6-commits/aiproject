
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import Holt

# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="월별 평균기온 분석 및 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 평균기온 분석 및 미래 예측")
st.write("1907년부터 2025년까지의 기온 데이터를 분석하고 미래 기온을 예측합니다.")

# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("temperature.csv", encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv("temperature.csv", encoding="utf-8-sig")

    df.columns = df.columns.str.strip()

    if "일시" not in df.columns:
        raise ValueError("'일시' 컬럼을 찾을 수 없습니다.")

    if "평균기온(℃)" not in df.columns:
        raise ValueError("'평균기온(℃)' 컬럼을 찾을 수 없습니다.")

    df["일시"] = pd.to_datetime(df["일시"], errors="coerce")
    df["평균기온(℃)"] = pd.to_numeric(
        df["평균기온(℃)"], errors="coerce"
    )

    df = df.dropna(subset=["일시"])
    df = df[
        (df["일시"].dt.year >= 1907) &
        (df["일시"].dt.year <= 2025)
    ].copy()

    df["연도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()

# --------------------------------------------------
# 3. 사용자 입력
# --------------------------------------------------

st.sidebar.header("⚙️ 분석 설정")

month = st.sidebar.selectbox(
    "📅 분석할 월 선택",
    list(range(1, 13)),
    format_func=lambda x: f"{x}월"
)

forecast_year = st.sidebar.number_input(
    "🔮 예측할 연도 입력",
    min_value=2026,
    max_value=2200,
    value=2030,
    step=1
)

st.sidebar.divider()

missing_method = st.sidebar.selectbox(
    "🧩 결측치 처리 방법",
    [
        "1. 제거",
        "2. 중앙값으로 대체",
        "3. 선형보간법"
    ]
)

algorithm = st.sidebar.selectbox(
    "🤖 예측 알고리즘 선택",
    [
        "선형회귀",
        "랜덤 포레스트",
        "홀트(Holt)"
    ]
)

# --------------------------------------------------
# 4. 월별 연도별 평균기온 데이터 만들기
# --------------------------------------------------

monthly_df = df[df["월"] == month].copy()

yearly_data = (
    monthly_df.groupby("연도")["평균기온(℃)"]
    .mean()
)

# 1907~2025년 전체 연도 생성
all_years = pd.Index(
    range(1907, 2026),
    name="연도"
)

yearly_data = yearly_data.reindex(all_years)

# --------------------------------------------------
# 5. 결측치 처리 함수
# --------------------------------------------------

def handle_missing(data, method):

    result = data.copy()

    if method == "1. 제거":
        result = result.dropna()

    elif method == "2. 중앙값으로 대체":
        median_value = result.median()

        if pd.isna(median_value):
            raise ValueError("대체할 중앙값이 없습니다.")

        result = result.fillna(median_value)

    elif method == "3. 선형보간법":
        result = result.interpolate(
            method="linear",
            limit_direction="both"
        )

    return result.dropna()


# --------------------------------------------------
# 6. 예측 알고리즘 함수
# --------------------------------------------------

def predict_temperature(data, target_year, model_name):

    data = data.dropna()

    if len(data) < 3:
        raise ValueError(
            "예측을 위해서는 최소 3개 이상의 데이터가 필요합니다."
        )

    years = data.index.to_numpy(dtype=float)
    values = data.to_numpy(dtype=float)

    future_years = np.arange(
        2026,
        target_year + 1
    )

    if len(future_years) == 0:
        return np.array([])

    # ① 선형회귀
    if model_name == "선형회귀":

        model = LinearRegression()

        model.fit(
            years.reshape(-1, 1),
            values
        )

        predictions = model.predict(
            future_years.reshape(-1, 1)
        )

    # ② 랜덤 포레스트
    elif model_name == "랜덤 포레스트":

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5
        )

        model.fit(
            years.reshape(-1, 1),
            values
        )

        predictions = model.predict(
            future_years.reshape(-1, 1)
        )

    # ③ 홀트
    elif model_name == "홀트(Holt)":

        model = Holt(
            values,
            initialization_method="estimated"
        )

        fitted_model = model.fit(optimized=True)

        steps = int(target_year - 2025)

        all_predictions = np.asarray(
            fitted_model.forecast(steps=steps),
            dtype=float
        )

        predictions = all_predictions[-len(future_years):]

    else:
        raise ValueError("올바르지 않은 알고리즘입니다.")

    return np.asarray(predictions, dtype=float)


# --------------------------------------------------
# 7. 데이터 결측치 처리
# --------------------------------------------------

try:
    selected_data = handle_missing(
        yearly_data,
        missing_method
    )

    if len(selected_data) < 3:
        st.error("선택한 월의 유효한 데이터가 부족합니다.")
        st.stop()

    main_predictions = predict_temperature(
        selected_data,
        int(forecast_year),
        algorithm
    )

except Exception as e:
    st.error(f"분석 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 8. 주요 정보 표시
# --------------------------------------------------

st.subheader(f"📊 {month}월 평균기온 분석 결과")

col1, col2, col3 = st.columns(3)

col1.metric(
    "분석 데이터 개수",
    f"{len(selected_data)}개"
)

col2.metric(
    "결측 데이터 개수",
    f"{int(yearly_data.isna().sum())}개"
)

col3.metric(
    "예측 연도",
    f"{forecast_year}년"
)

st.caption(
    f"선택한 처리 방법: {missing_method} | "
    f"선택한 알고리즘: {algorithm}"
)

# --------------------------------------------------
# 9. 실제 기온 + 미래 예측 그래프
# --------------------------------------------------

st.divider()

st.subheader("🌡️ 실제 기온과 미래 예측 그래프")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=selected_data.index,
    y=selected_data.values,
    mode="lines",
    name="실제 평균기온",
    line=dict(width=2)
))

future_years = np.arange(2026, forecast_year + 1)

fig.add_trace(go.Scatter(
    x=future_years,
    y=main_predictions,
    mode="lines+markers",
    name=f"{algorithm} 예측",
    line=dict(width=3, dash="dash")
))

fig.update_layout(
    title=f"{month}월 평균기온: 1907~{forecast_year}년",
    xaxis_title="연도",
    yaxis_title="평균기온(℃)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

if len(main_predictions) > 0:
    st.success(
        f"🔮 {forecast_year}년 {month}월 예측 평균기온: "
        f"{main_predictions[-1]:.2f}℃"
    )

# --------------------------------------------------
# 10. 결측치 처리 방법별 예측 비교
# --------------------------------------------------

st.divider()

st.subheader("🧩 결측치 처리 방법에 따른 예측 결과 비교")

st.write(
    "동일한 알고리즘을 사용하고 결측치 처리 방법만 변경하여 "
    "예측 결과를 비교합니다."
)

methods = [
    "1. 제거",
    "2. 중앙값으로 대체",
    "3. 선형보간법"
]

missing_fig = go.Figure()

missing_results = {}

for method in methods:

    try:
        processed_data = handle_missing(
            yearly_data,
            method
        )

        predictions = predict_temperature(
            processed_data,
            int(forecast_year),
            algorithm
        )

        missing_results[method] = predictions

        missing_fig.add_trace(go.Scatter(
            x=future_years,
            y=predictions,
            mode="lines+markers",
            name=method
        ))

    except Exception as e:
        st.warning(f"{method} 처리 실패: {e}")

missing_fig.update_layout(
    title=f"{algorithm} 알고리즘 - 결측치 처리 방법별 예측",
    xaxis_title="연도",
    yaxis_title="예측 평균기온(℃)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(
    missing_fig,
    use_container_width=True
)

# 예측값 비교표
if missing_results:

    comparison_data = []

    for method, predictions in missing_results.items():

        if len(predictions) > 0:
            comparison_data.append({
                "결측치 처리 방법": method,
                f"{forecast_year}년 예측 기온(℃)": round(
                    float(predictions[-1]), 2
                )
            })

    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 11. 알고리즘별 예측 결과 비교
# --------------------------------------------------

st.divider()

st.subheader("🤖 알고리즘에 따른 예측 결과 비교")

st.write(
    "동일한 결측치 처리 방법을 적용하고 "
    "예측 알고리즘만 변경하여 비교합니다."
)

algorithms = [
    "선형회귀",
    "랜덤 포레스트",
    "홀트(Holt)"
]

algorithm_fig = go.Figure()

algorithm_results = {}

for model_name in algorithms:

    try:
        predictions = predict_temperature(
            selected_data,
            int(forecast_year),
            model_name
        )

        algorithm_results[model_name] = predictions

        algorithm_fig.add_trace(go.Scatter(
            x=future_years,
            y=predictions,
            mode="lines+markers",
            name=model_name
        ))

    except Exception as e:
        st.warning(f"{model_name} 예측 실패: {e}")

algorithm_fig.update_layout(
    title=f"{missing_method} 적용 - 알고리즘별 예측",
    xaxis_title="연도",
    yaxis_title="예측 평균기온(℃)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(
    algorithm_fig,
    use_container_width=True
)

# 알고리즘별 예측값 표
if algorithm_results:

    algorithm_table = []

    for name, predictions in algorithm_results.items():

        if len(predictions) > 0:
            algorithm_table.append({
                "예측 알고리즘": name,
                f"{forecast_year}년 예측 기온(℃)": round(
                    float(predictions[-1]), 2
                )
            })

    st.dataframe(
        pd.DataFrame(algorithm_table),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 12. 결측치 처리 전후 데이터 확인
# --------------------------------------------------

st.divider()

with st.expander("🔍 결측치 처리 전후 데이터 확인하기"):

    before_df = yearly_data.rename(
        "처리 전 평균기온"
    ).reset_index()

    after_df = selected_data.rename(
        "처리 후 평균기온"
    ).reset_index()

    st.write("결측치 처리 전")
    st.dataframe(
        before_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("결측치 처리 후")
    st.dataframe(
        after_df,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "※ 예측값은 과거 데이터에 기반한 통계적 추정치이며 "
    "실제 미래 기온과 다를 수 있습니다."
)
