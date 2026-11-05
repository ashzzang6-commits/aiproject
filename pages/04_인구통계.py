
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re

st.set_page_config(
    page_title="인구통계 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 대한민국 인구통계 분석")
st.write("2025년 연령별 인구 분포를 확인해 보세요!")

# CSV 파일 경로
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "population.csv"


# 데이터 불러오기
@st.cache_data
def load_data():
    for encoding in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(CSV_PATH, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 확인해 주세요.")


if not CSV_PATH.exists():
    st.error("population.csv 파일을 찾을 수 없습니다.")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 오류: {e}")
    st.stop()


# 컬럼 이름 정리
df.columns = df.columns.astype(str).str.strip()

if "행정구역" not in df.columns:
    st.error("행정구역 컬럼이 없습니다.")
    st.stop()

# 지역명 정리
df["지역명"] = (
    df["행정구역"]
    .astype(str)
    .str.split(" (", regex=False)
    .str[0]
    .str.strip()
)

# 성별 선택
st.subheader("🔎 분석 조건 선택")

col1, col2 = st.columns(2)

with col1:
    regions = sorted(df["지역명"].dropna().unique())
    selected_region = st.selectbox(
        "행정구역 선택",
        regions
    )

with col2:
    selected_gender = st.selectbox(
        "성별 선택",
        ["전체", "남성", "여성"]
    )


# 성별에 따른 컬럼 접두사
gender_prefix = {
    "전체": "2025년_거주자_",
    "남성": "2025년_남_거주자_",
    "여성": "2025년_여_거주자_"
}

prefix = gender_prefix[selected_gender]

# 연령별 컬럼 찾기
age_columns = {}

for column in df.columns:
    if column.startswith(prefix):
        age_text = column.replace(prefix, "", 1)

        match = re.fullmatch(
            r"(\d+)~(\d+)세",
            age_text
        )

        if match:
            age = int(match.group(1))
            age_columns[age] = column

        elif age_text == "100세 이상":
            age_columns[100] = column


if not age_columns:
    st.error("연령별 인구 컬럼을 찾을 수 없습니다.")
    st.write("현재 컬럼:", list(df.columns))
    st.stop()


# 선택 지역 데이터
region_data = df[df["지역명"] == selected_region]

if region_data.empty:
    st.warning("선택한 지역의 데이터가 없습니다.")
    st.stop()


# 연령별 인구 데이터 계산
result = []

for age in sorted(age_columns):
    column = age_columns[age]

    population = pd.to_numeric(
        region_data[column].astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0).sum()

    result.append({
        "나이": age,
        "인구수": int(population)
    })

chart_df = pd.DataFrame(result)

# 전체 인구 합계
total_population = chart_df["인구수"].sum()

st.divider()

# 인구 정보
col1, col2 = st.columns(2)

with col1:
    st.metric("📍 선택 지역", selected_region)

with col2:
    st.metric("👥 연령별 인구 합계", f"{total_population:,}명")


# 그래프 제목
st.subheader(
    f"📈 {selected_region} 연령별 인구 분포"
)

# 그래프 생성
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_df["나이"],
        y=chart_df["인구수"],
        mode="lines+markers",
        name="인구수",
        line=dict(
            color="#FF69B4",
            width=3
        ),
        marker=dict(
            color="#FF69B4",
            size=8
        )
    )
)

# Y축 범위
max_value = int(chart_df["인구수"].max())
y_max = max(
    1000000,
    ((max_value // 1000000) + 1) * 1000000
)

fig.update_layout(
    height=550,
    template="plotly_white",
    xaxis=dict(
        title="나이 (연령대 시작 나이)",
        tickmode="linear",
        tick0=0,
        dtick=10
    ),
    yaxis=dict(
        title="인구수 (명)",
        tickmode="linear",
        tick0=0,
        dtick=1000000,
        range=[0, y_max]
    ),
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# 데이터 표
with st.expander("📋 연령별 인구 데이터 보기"):
    display_df = chart_df.copy()
    display_df["인구수"] = display_df["인구수"].map(
        lambda x: f"{x:,}명"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

st.success("인구통계 분석이 완료되었습니다! 🎉")

