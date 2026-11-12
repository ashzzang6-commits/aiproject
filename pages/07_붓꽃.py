
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스 꽃 품종 예측")
st.write("꽃잎의 길이와 너비를 입력하면 아이리스 품종을 예측해 줘!")

# CSV 파일 경로: pages 폴더의 상위 폴더
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "iris.csv")

# 데이터 불러오기
@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        
        # CSV 컬럼명이 다를 경우 기본 아이리스 데이터 사용
        if len(df.columns) >= 5:
            return df

    # CSV 파일이 없으면 sklearn 기본 데이터 사용
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    df["species"] = [
        iris.target_names[i] for i in iris.target
    ]
    return df

try:
    df = load_data()

    st.success("✅ 아이리스 데이터를 불러왔어요!")

    st.subheader("🌷 꽃잎 정보 입력")

    col1, col2 = st.columns(2)

    with col1:
        petal_length = st.number_input(
            "꽃잎 길이 (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1
        )

    with col2:
        petal_width = st.number_input(
            "꽃잎 너비 (cm)",
            min_value=0.0,
            max_value=5.0,
            value=0.3,
            step=0.1
        )

    # 데이터 컬럼 설정
    feature_length = "petal length (cm)"
    feature_width = "petal width (cm)"
    target_column = "species"

    # CSV 컬럼 자동 확인
    if feature_length not in df.columns:
        feature_length = df.columns[2]

    if feature_width not in df.columns:
        feature_width = df.columns[3]

    if target_column not in df.columns:
        target_column = df.columns[-1]

    # 모델 학습
    X = df[[feature_length, feature_width]]
    y = df[target_column]

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    # 예측 버튼
    if st.button("🌼 품종 예측하기", use_container_width=True):

        prediction = model.predict(
            [[petal_length, petal_width]]
        )

        st.success(f"예측된 품종은 **{prediction[0]}** 입니다!")

    # 그래프
    st.subheader("📊 아이리스 꽃 데이터 시각화")

    fig, ax = plt.subplots(figsize=(9, 5))

    for species in df[target_column].unique():
        subset = df[df[target_column] == species]

        ax.scatter(
            subset[feature_length],
            subset[feature_width],
            label=str(species),
            alpha=0.7
        )

    ax.scatter(
        petal_length,
        petal_width,
        color="red",
        marker="*",
        s=250,
        label="입력한 꽃"
    )

    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.set_title("Iris Petal Data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")


