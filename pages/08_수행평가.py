
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import zipfile
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="자동차 데이터 분석",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------
# 디자인
# ---------------------------
st.markdown("""
<style>
.stApp {
    background-color: #101827;
}
h1, h2, h3 {
    color: #ffffff;
}
[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 USED CAR DATA ANALYTICS")
st.caption("중고차 시장 분석 및 AI 가격 예측 시스템")

# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data(show_spinner="자동차 데이터를 분석하는 중입니다...")
def load_data():

    file_path = Path(__file__).resolve().parent.parent / "cars.zip"

    if not file_path.exists():
        raise FileNotFoundError("상위 폴더에서 cars.zip을 찾을 수 없습니다.")

    columns = [
        "price", "year", "manufacturer", "model",
        "condition", "cylinders", "fuel",
        "odometer", "transmission", "drive",
        "type", "state"
    ]

    samples = []

    with zipfile.ZipFile(file_path) as z:
        csv_name = next(
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        )

        with z.open(csv_name) as f:
            for chunk in pd.read_csv(
                f,
                usecols=columns,
                chunksize=100000,
                low_memory=True,
                on_bad_lines="skip"
            ):
                chunk = chunk.sample(
                    n=min(3000, len(chunk)),
                    random_state=42
                )
                samples.append(chunk)

    df = pd.concat(samples, ignore_index=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(
        df["odometer"], errors="coerce"
    )

    df = df.dropna(subset=["price", "year", "odometer"])

    df = df[
        (df["price"] >= 500) &
        (df["price"] <= 200000) &
        (df["year"] >= 1980) &
        (df["year"] <= 2027) &
        (df["odometer"] >= 0) &
        (df["odometer"] <= 400000)
    ].copy()

    for col in [
        "manufacturer", "model", "condition",
        "fuel", "transmission", "drive", "type", "state"
    ]:
        df[col] = df[col].fillna("unknown").astype(str).str.lower()

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 로딩 오류: {e}")
    st.stop()

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()

# ---------------------------
# 사이드바 필터
# ---------------------------
st.sidebar.title("⚙️ 분석 설정")

manufacturers = sorted(df["manufacturer"].unique())

selected_manufacturers = st.sidebar.multiselect(
    "자동차 제조사",
    manufacturers,
    default=manufacturers[:min(5, len(manufacturers))]
)

fuel_types = sorted(df["fuel"].unique())

selected_fuel = st.sidebar.multiselect(
    "연료 종류",
    fuel_types,
    default=fuel_types
)

year_min = int(df["year"].min())
year_max = int(df["year"].max())

selected_year = st.sidebar.slider(
    "자동차 연식",
    year_min,
    year_max,
    (year_min, year_max)
)

filtered = df[
    df["manufacturer"].isin(selected_manufacturers) &
    df["fuel"].isin(selected_fuel) &
    df["year"].between(*selected_year)
]

if filtered.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다.")
    st.stop()

# ---------------------------
# 핵심 통계
# ---------------------------
st.header("📊 시장 현황")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "분석 차량 수",
    f"{len(filtered):,}대"
)

c2.metric(
    "평균 판매 가격",
    f"${filtered['price'].mean():,.0f}"
)

c3.metric(
    "평균 주행거리",
    f"{filtered['odometer'].mean():,.0f} mi"
)

c4.metric(
    "평균 연식",
    f"{filtered['year'].mean():.1f}년"
)

st.divider()

# ---------------------------
# 탭 구성
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 가격 분석",
    "🏭 제조사 분석",
    "🚘 자동차 특성",
    "🤖 AI 가격 예측"
])

# ---------------------------
# TAB 1 가격 분석
# ---------------------------
with tab1:

    st.subheader("중고차 가격 분포")

    fig = px.histogram(
        filtered,
        x="price",
        nbins=50,
        title="자동차 가격 분포",
        labels={"price": "가격 (USD)"},
        color_discrete_sequence=["#38bdf8"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("연식과 가격의 관계")

    year_price = (
        filtered.groupby("year", as_index=False)
        ["price"].median()
    )

    fig = px.line(
        year_price,
        x="year",
        y="price",
        markers=True,
        title="연식별 중간 가격",
        labels={
            "year": "자동차 연식",
            "price": "중간 가격 (USD)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("주행거리와 가격")

    scatter_data = filtered.sample(
        min(3000, len(filtered)),
        random_state=42
    )

    fig = px.scatter(
        scatter_data,
        x="odometer",
        y="price",
        color="condition",
        title="주행거리에 따른 가격 변화",
        labels={
            "odometer": "주행거리 (마일)",
            "price": "가격 (USD)"
        },
        opacity=0.6
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 2 제조사 분석
# ---------------------------
with tab2:

    st.subheader("제조사별 평균 가격")

    manufacturer_price = (
        filtered.groupby("manufacturer")
        .agg(
            평균가격=("price", "mean"),
            차량수=("price", "count")
        )
        .reset_index()
        .sort_values("평균가격", ascending=False)
        .head(15)
    )

    fig = px.bar(
        manufacturer_price,
        x="manufacturer",
        y="평균가격",
        color="평균가격",
        title="제조사별 평균 판매 가격",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("제조사별 판매 차량 수")

    fig = px.pie(
        manufacturer_price,
        names="manufacturer",
        values="차량수",
        hole=0.4,
        title="주요 제조사 차량 비율"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        manufacturer_price,
        use_container_width=True
    )

# ---------------------------
# TAB 3 자동차 특성
# ---------------------------
with tab3:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("연료 종류별 차량 수")

        fuel_count = (
            filtered["fuel"]
            .value_counts()
            .reset_index()
        )

        fuel_count.columns = ["연료", "차량수"]

        fig = px.pie(
            fuel_count,
            names="연료",
            values="차량수",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("차종별 차량 수")

        type_count = (
            filtered["type"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        type_count.columns = ["차종", "차량수"]

        fig = px.bar(
            type_count,
            x="차종",
            y="차량수",
            color="차량수"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("변속기별 평균 가격")

    trans_price = (
        filtered.groupby("transmission")["price"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        trans_price,
        x="transmission",
        y="price",
        color="transmission",
        title="변속기별 평균 자동차 가격"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("상세 자동차 데이터")

    st.dataframe(
        filtered.head(500),
        use_container_width=True
    )

# ---------------------------
# TAB 4 AI 가격 예측
# ---------------------------
with tab4:

    st.subheader("🤖 AI 중고차 가격 예측")

    st.write(
        "자동차의 연식, 주행거리, 제조사 등의 정보를 "
        "입력하면 머신러닝으로 예상 가격을 계산합니다."
    )

    train_data = df.sample(
        min(10000, len(df)),
        random_state=42
    ).copy()

    numeric_features = ["year", "odometer"]

    categorical_features = [
        "manufacturer",
        "fuel",
        "transmission",
        "type",
        "condition"
    ]

    X = train_data[
        numeric_features + categorical_features
    ]

    y = train_data["price"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            random_state=42,
            n_jobs=-1
        ))
    ])

    st.info("예측 모델은 최대 10,000대의 표본 데이터로 학습합니다.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            input_year = st.number_input(
                "자동차 연식",
                min_value=1980,
                max_value=2027,
                value=2020
            )

            input_odometer = st.number_input(
                "주행거리 (마일)",
                min_value=0,
                max_value=400000,
                value=50000
            )

            input_manufacturer = st.selectbox(
                "제조사",
                sorted(df["manufacturer"].unique())
            )

            input_fuel = st.selectbox(
                "연료",
                sorted(df["fuel"].unique())
            )

        with col2:

            input_transmission = st.selectbox(
                "변속기",
                sorted(df["transmission"].unique())
            )

            input_type = st.selectbox(
                "차종",
                sorted(df["type"].unique())
            )

            input_condition = st.selectbox(
                "차량 상태",
                sorted(df["condition"].unique())
            )

        submitted = st.form_submit_button(
            "🚀 예상 가격 분석하기",
            use_container_width=True
        )

    if submitted:

        with st.spinner("AI 모델 학습 및 가격 예측 중..."):

            try:
                model.fit(X, y)

                input_data = pd.DataFrame([{
                    "year": input_year,
                    "odometer": input_odometer,
                    "manufacturer": input_manufacturer,
                    "fuel": input_fuel,
                    "transmission": input_transmission,
                    "type": input_type,
                    "condition": input_condition
                }])

                prediction = model.predict(input_data)[0]

                st.success("가격 예측이 완료되었습니다!")

                st.metric(
                    "AI 예상 중고차 가격",
                    f"${prediction:,.0f}"
                )

                st.caption(
                    "예측값은 표본 데이터에 기반한 추정치이며 "
                    "실제 거래 가격과 다를 수 있습니다."
                )

            except Exception as e:
                st.error(f"예측 오류: {e}")

# ---------------------------
# 하단
# ---------------------------
st.divider()

st.caption(
    "USED CAR DATA ANALYTICS SYSTEM | "
    "Powered by Streamlit & Machine Learning"
)

