
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# 1. 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="서울시 연령별 인구 분석",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# 2. CSV 파일 경로 설정
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "population.csv"

# -----------------------------
# 3. CSV 파일 불러오기
# -----------------------------
@st.cache_data
def load_data():
    encodings = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                CSV_PATH,
                encoding=encoding
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            "CSV 파일의 인코딩을 확인해 주세요."
        )

    # 인구수에 포함된 쉼표 제거 후 숫자로 변환
    for col in df.columns:
        if col.startswith("2025년_"):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            )

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "population.csv 파일을 찾을 수 없습니다. "
        "상위 폴더에 CSV 파일이 있는지 확인해 주세요."
    )
    st.stop()

# -----------------------------
# 4. 제목
# -----------------------------
st.title("🌸 서울시 연령별 인구 분석")
st.write(
    "행정구역과 성별을 선택해 "
    "연령대별 인구를 꺾은선 그래프로 확인해 보세요."
)

# -----------------------------
# 5. 행정구역 이름 정리
# -----------------------------
df["지역명"] = (
    df["행정구역"]
    .str.split(" (")
    .str[0]
    .str.strip()
)

# -----------------------------
# 6. 선택 메뉴
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    selected_region = st.selectbox(
        "📍 행정구역 선택",
        df["지역명"].tolist()
    )

with col2:
    selected_gender = st.selectbox(
        "🚻 성별 선택",
        ["전체", "남성", "여성"]
    )

# -----------------------------
# 7. 선택한 데이터 찾기
# -----------------------------
selected_data = df[
    df["지역명"] == selected_region
].iloc[0]

# -----------------------------
# 8. 성별에 따른 데이터 열 선택
# -----------------------------
if selected_gender == "전체":
    prefix = "2025년_거주자_"
elif selected_gender == "남성":
    prefix = "2025년_남_거주자_"
else:
    prefix = "2025년_여_거주자_"

# -----------------------------
# 9. 연령대별 인구 데이터
# -----------------------------
age_labels = [
    "0~9세",
    "10~19세",
    "20~29세",
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상"
]

age_columns = [
    prefix + "0~9세",
    prefix + "10~19세",
    prefix + "20~29세",
    prefix + "30~39세",
    prefix + "40~49세",
    prefix + "50~59세",
    prefix + "60~69세",
    prefix + "70~79세",
    prefix + "80~89세",
    prefix + "90~99세",
    prefix + "100세 이상"
]

population = [
    int(selected_data[col])
    for col in age_columns
]

# -----------------------------
# 10. 꺾은선 그래프
# -----------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=age_labels,
        y=population,
        mode="lines+markers",
        name="인구수",
        line=dict(
            color="#FF1493",
            width=4
        ),
        marker=dict(
            color="#FF1493",
            size=9
        ),
        hovertemplate=(
            "연령대: %{x}<br>"
            "인구수: %{y:,.0f}명"
            "<extra></extra>"
        )
    )
)

# -----------------------------
# 11. 그래프 디자인
# -----------------------------
fig.update_layout(
    title=f"{selected_region} {selected_gender} 연령별 인구",
    xaxis=dict(
        title="나이",
        type="category",
        categoryorder="array",
        categoryarray=age_labels,
        tickmode="array",
        tickvals=age_labels,
        ticktext=age_labels,
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=1
    ),
    yaxis=dict(
        title="인구수 (명)",
        rangemode="tozero",
        tickmode="linear",
        tick0=0,
        dtick=1000000,
        tickformat=",",
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=1
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    height=550,
    margin=dict(
        l=30,
        r=30,
        t=70,
        b=30
    )
)

# -----------------------------
# 12. 그래프 출력
# -----------------------------
st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# 13. 선택한 데이터 표시
# -----------------------------
st.subheader("📊 연령대별 인구 데이터")

result = pd.DataFrame({
    "연령대": age_labels,
    "인구수": population
})

result["인구수"] = result["인구수"].map(
    lambda x: f"{x:,}명"
)

st.dataframe(
    result,
    hide_index=True,
    use_container_width=True
)


   
  
