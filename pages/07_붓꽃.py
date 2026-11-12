
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# 페이지 설정
st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌷",
    layout="wide"
)

st.title("🌷 아이리스 붓꽃 품종 예측")
st.write("꽃잎의 길이와 너비를 입력하면 머신러닝이 붓꽃의 품종을 예측해 줍니다!")

st.divider()

# --------------------------------
# 1. 데이터 불러오기
# --------------------------------

@st.cache_data
def load_data():
    folder = Path(__file__).parent

    candidates = [
        folder / "iris.csv",
        folder / "아이리스.csv"
    ]

    file_path = next(
        (p for p in candidates if p.exists()),
        None
    )

    if file_path is None:
        raise FileNotFoundError(
            "iris.csv 또는 아이리스.csv 파일을 찾을 수 없습니다."
        )

    # 한글 CSV 인코딩 대응
    df = None

    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("CSV 파일의 인코딩을 확인해 주세요.")

    # 업로드한 데이터의 열 이름이 잘못 합쳐진 경우를 대비해
    # 5개 열을 위치 기준으로 정리
    if len(df.columns) != 5:
        raise ValueError("아이리스 데이터는 5개 열이어야 합니다.")

    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    df["petal_length"] = pd.to_numeric(
        df["petal_length"], errors="coerce"
    )
    df["petal_width"] = pd.to_numeric(
        df["petal_width"], errors="coerce"
    )

    df = df.dropna(
        subset=["petal_length", "petal_width", "species"]
    )

    df["species"] = df["species"].astype(str).str.strip()

    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------
# 2. 품종 이름
# --------------------------------

species_names = {
    "Iris-setosa": "Setosa",
    "Iris-versicolor": "Versicolor",
    "Iris-virginica": "Virginica",
    "setosa": "Setosa",
    "versicolor": "Versicolor",
    "virginica": "Virginica"
}

df["품종"] = df["species"].map(
    lambda x: species_names.get(x, x)
)

# --------------------------------
# 3. 머신러닝 모델 생성
# --------------------------------

X = df[["petal_length", "petal_width"]]
y = df["품종"]

# 표준화 + KNN 분류
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

model.fit(X, y)

# 교차검증 정확도
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

# --------------------------------
# 4. 사용자 입력
# --------------------------------

st.subheader("✏️ 꽃잎 크기 입력하기")

col1, col2 = st.columns(2)

with col1:
    petal_length = st.number_input(
        "🌸 꽃잎 길이 (cm)",
        min_value=0.1,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:
    petal_width = st.number_input(
        "🌸 꽃잎 너비 (cm)",
        min_value=0.1,
        max_value=5.0,
        value=0.2,
        step=0.1
    )

# --------------------------------
# 5. 품종 예측
# --------------------------------

input_data = pd.DataFrame(
    [[petal_length, petal_width]],
    columns=["petal_length", "petal_width"]
)

prediction = model.predict(input_data)[0]

# 최근접 이웃 5개 찾기
scaled_X = model.named_steps["scaler"].transform(X)
scaled_input = model.named_steps["scaler"].transform(input_data)

distances, indices = model.named_steps["knn"].kneighbors(
    scaled_input,
    n_neighbors=5
)

nearest = df.iloc[indices[0]].copy()
nearest["거리"] = distances[0]

# 품종별 투표 수
vote_counts = nearest["품종"].value_counts()

# --------------------------------
# 6. 예측 결과
# --------------------------------

st.divider()
st.subheader("🔍 머신러닝 예측 결과")

st.success(f"🎉 예측된 붓꽃 품종은 {prediction} 입니다!")

result_col1, result_col2 = st.columns(2)

with result_col1:
    st.metric(
        "예측 품종",
        prediction
    )

with result_col2:
    st.metric(
        "5개 이웃의 교차검증 평균 정확도",
        f"{cv_scores.mean() * 100:.1f}%"
    )

st.caption(
    "정확도는 5겹 교차검증 결과이며, 입력한 꽃의 개별 예측 확률을 의미하지 않습니다."
)

# --------------------------------
# 7. 그래프 시각화
# --------------------------------

st.divider()
st.subheader("📊 꽃잎 길이와 너비에 따른 품종 분포")

fig = go.Figure()

color_map = {
    "Setosa": "#FF9999",
    "Versicolor": "#66B2FF",
    "Virginica": "#66CC99"
}

# 기존 데이터 품종별 표시
for species in df["품종"].unique():

    subset = df[df["품종"] == species]

    fig.add_trace(
        go.Scatter(
            x=subset["petal_length"],
            y=subset["petal_width"],
            mode="markers",
            name=species,
            marker=dict(
                size=9,
                color=color_map.get(species, "#999999"),
                opacity=0.7
            ),
            hovertemplate=(
                "품종: " + species +
                "<br>꽃잎 길이: %{x} cm" +
                "<br>꽃잎 너비: %{y} cm" +
                "<extra></extra>"
            )
        )
    )

# 가장 가까운 5개 데이터 연결
for _, row in nearest.iterrows():

    fig.add_trace(
        go.Scatter(
            x=[petal_length, row["petal_length"]],
            y=[petal_width, row["petal_width"]],
            mode="lines",
            line=dict(
                color="rgba(100,100,100,0.35)",
                dash="dot",
                width=1.5
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

# 최근접 이웃 5개 강조
fig.add_trace(
    go.Scatter(
        x=nearest["petal_length"],
        y=nearest["petal_width"],
        mode="markers",
        name="가까운 데이터 5개",
        marker=dict(
            symbol="circle-open",
            size=17,
            color="#222222",
            line=dict(width=2)
        ),
        hovertemplate=(
            "품종: %{customdata[0]}" +
            "<br>거리: %{customdata[1]:.3f}" +
            "<extra></extra>"
        ),
        customdata=np.column_stack([
            nearest["품종"],
            nearest["거리"]
        ])
    )
)

# 사용자가 입력한 값 (보라색 별)
fig.add_trace(
    go.Scatter(
        x=[petal_length],
        y=[petal_width],
        mode="markers",
        name="내가 입력한 값",
        marker=dict(
            symbol="star",
            size=25,
            color="#A020F0",
            line=dict(
                color="white",
                width=1.5
            )
        ),
        hovertemplate=(
            "내가 입력한 값" +
            "<br>꽃잎 길이: %{x} cm" +
            "<br>꽃잎 너비: %{y} cm" +
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="꽃잎 길이 (cm)",
    yaxis_title="꽃잎 너비 (cm)",
    height=600,
    template="plotly_white",
    legend_title="범례",
    hovermode="closest"
)

fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "⭐ 보라색 별: 사용자 입력값 | "
    "검은색 테두리 원: 가장 가까운 데이터 5개 | "
    "점선: 입력값과 가까운 데이터의 거리"
)

# --------------------------------
# 8. 예측 이유 분석
# --------------------------------

st.divider()
st.subheader("🧠 왜 이 품종으로 예측되었을까?")

st.write(
    "KNN 알고리즘은 입력값과 가장 가까운 데이터 5개를 찾아 "
    "그중 가장 많이 등장한 품종을 선택합니다."
)

# 투표 결과 그래프
vote_df = pd.DataFrame({
    "품종": ["Setosa", "Versicolor", "Virginica"],
    "투표 수": [
        vote_counts.get("Setosa", 0),
        vote_counts.get("Versicolor", 0),
        vote_counts.get("Virginica", 0)
    ]
})

vote_fig = px.bar(
    vote_df,
    x="품종",
    y="투표 수",
    color="품종",
    color_discrete_map=color_map,
    text="투표 수",
    title="가장 가까운 데이터 5개의 품종 투표 결과"
)

vote_fig.update_layout(
    showlegend=False,
    yaxis=dict(
        tickmode="linear",
        dtick=1,
        range=[0, 5]
    ),
    template="plotly_white"
)

st.plotly_chart(
    vote_fig,
    use_container_width=True
)

# 최근접 이웃 상세 정보
st.write("**🔎 가장 가까운 데이터 5개**")

detail = nearest[
    ["petal_length", "petal_width", "품종", "거리"]
].copy()

detail.columns = [
    "꽃잎 길이",
    "꽃잎 너비",
    "품종",
    "표준화 거리"
]

detail["표준화 거리"] = detail["표준화 거리"].round(3)

st.dataframe(
    detail.reset_index(drop=True),
    use_container_width=True
)

winner_count = vote_counts.get(prediction, 0)

st.info(
    f"가장 가까운 데이터 5개 중 {winner_count}개가 "
    f"{prediction} 품종입니다. "
    f"따라서 KNN 알고리즘은 {prediction}으로 분류했습니다."
)

# --------------------------------
# 9. 알고리즘 설명
# --------------------------------

with st.expander("📚 KNN 알고리즘 알아보기"):

    st.write("""
    **KNN (K-Nearest Neighbors)**

    1. 기존 붓꽃 데이터를 학습합니다.
    2. 사용자가 꽃잎 길이와 너비를 입력합니다.
    3. 기존 데이터와 입력값 사이의 거리를 계산합니다.
    4. 가장 가까운 5개의 데이터를 선택합니다.
    5. 가장 많이 나온 품종을 최종 결과로 예측합니다.

    이번 프로그램은 꽃잎 길이와 너비를
    StandardScaler로 표준화한 뒤 거리를 계산합니다.

    K값은 5로 설정했습니다.
    """)

st.divider()
st.caption("🌷 Iris Machine Learning Project | Streamlit + Pandas + Scikit-learn")
