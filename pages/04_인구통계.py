
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="인구통계 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 대한민국 인구통계 분석")
st.write("행정구역과 성별에 따른 연령별 인구를 확인해 보세요!")

# CSV 파일 경로
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "population.csv"


# CSV 파일 읽기
@st.cache_data
def load_data(path):
    encodings = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                dtype=str
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 한글 인코딩을 확인해 주세요.")


# 파일 존재 확인
if not CSV_PATH.exists():
    st.error("population.csv 파일을 찾을 수 없습니다.")
    st.info("population.csv 파일이 프로젝트 최상위 폴더에 있는지 확인해 주세요.")
    st.stop()

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


# 컬럼 이름 정리
df.columns = df.columns.str.strip()

# 행정구역 컬럼 확인
if "행정구역" not in df.columns:
    st.error("CSV 파일에 '행정구역' 컬럼이 없습니다.")
    st.write("현재 CSV 컬럼:", list(df.columns))
    st.stop()


# 행정구역 이름 정리
df["지역명"] = (
    df["행정구역"]
    .fillna("")
    .astype(str)
    .str.split(" (", regex=False)
    .str[0]
    .str.strip()
)

df = df[df["지역명"] != ""]


# 성별별 연령 컬럼 찾기
age_columns = {
    "전체": {},
    "남성": {},
    "여성": {}
}

for col in df.columns:
    col_name = str(col).strip()

    # 계, 남자, 여자 뒤에 있는 나이 찾기
    match = re.search(
        r"_(계|남자|여자)_(\d+)세$",
        col_name
    )

    # 100세 이상 처리
    if match is None:
        match = re.search(
            r"_(계|남자|여자)_(100세 이상)$",
            col_name
        )

    if match:
        gender = match.group(1)
        age_text = match.group(2)

        if age_text == "100세 이상":
            age = 100
        else:
            age = int(age_text.replace("세", ""))

        if gender == "계":
            category = "전체"
        elif gender == "남자":
            category = "남성"
        else:
            category = "여성"

        age_columns[category][age] = col


# 연령 데이터 확인
if not any(age_columns.values()):
    st.error("연령별 인구 컬럼을 찾을 수 없습니다.")
    st.write("CSV 컬럼 목록:")
    st.write(list(df.columns))
    st.stop()


# 선택 영역
st.subheader("🔎 분석 조건 선택")

col1, col2 = st.columns(2)

with col1:
    regions = sorted(df["지역명"].unique().tolist())

    selected_region = st.selectbox(
        "행정구역 선택",
        regions
    )

with col2:
    selected_gender = st.selectbox(
        "성별 선택",
        ["전체", "남성", "여성"]
    )


# 선택한 지역 데이터
selected_data = df[df["지역명"] == selected_region]

if selected_data.empty:
    st.warning("선택한 행정구역의 데이터가 없습니다.")
    st.stop()


# 선택한 성별의 연령 컬럼
selected_columns = age_columns[selected_gender]

if not selected_columns:
    st.error(
        f"{selected_gender}의 연령별 인구 데이터가 없습니다."
    )
    st.stop()


# 연령별 인구 데이터 만들기
result = []

for age, column in sorted(selected_columns.items()):
    values = (
        selected_data[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    population = pd.to_numeric(
        values,
        errors="coerce"
    ).fillna(0).sum()

    result.append({
        "나이": age,
        "인구수": int(population)
    })

chart_df = pd.DataFrame(result)

if chart_df.empty:
    st.warning("그래프를 표시할 데이터가 없습니다.")
    st.stop()


# 인구 정보
total_population = int(chart_df["인구수"].sum())

st.divider()

metric1, metric2 = st.columns(2)

with metric1:
    st.metric(
        "선택한 행정구역",
        selected_region
    )

with metric2:
    st.metric(
        "연령별 인구 합계",
        f"{total_population:,}명"
    )


# 그래프
st.subheader(
    f"📈 {selected_region} 연령별 인구 분포 ({selected_gender})"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_df["나이"],
        y=chart_df["인구수"],
        mode="lines+markers",
        name="인구수",
        line=dict(
            color="hotpink",
            width=3
        ),
        marker=dict(
            size=6,
            color="hotpink"
        )
    )
)

# Y축 눈금 계산
max_population = int(chart_df["인구수"].max())
y_max = max(
    1000000,
    ((max_population // 1000000) + 1) * 1000000
)

fig.update_layout(
    xaxis_title="나이",
    yaxis_title="인구수 (명)",
    template="plotly_white",
    height=550,
    hovermode="x unified",
    xaxis=dict(
        tickmode="linear",
        tick0=0,
        dtick=10
    ),
    yaxis=dict(
        tickmode="linear",
        tick0=0,
        dtick=1000000,
        range=[0, y_max]
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# 데이터 표
with st.expander("📋 연령별 인구 데이터 확인하기"):
    display_df = chart_df.copy()
    display_df["인구수"] = display_df["인구수"].map(
        lambda x: f"{x:,}명"
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

st.caption("※ 제공된 population.csv 데이터를 기준으로 분석합니다.")

  
