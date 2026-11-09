
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="월별 기온분석",
    page_icon="🌡️",
    layout="wide"
)

# 제목
st.title("🌡️ 월별 기온분석")
st.write("원하는 월을 선택하면 연도별 최고기온과 최저기온을 비교할 수 있어요!")

# CSV 파일 불러오기 (한글 깨짐 방지)
@st.cache_data
def load_data():
    df = pd.read_csv("temperature.csv", encoding="cp949")

    # 날짜에서 연도와 월 추출
    month_dict = {
        "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,
        "Oct": 10, "Nov": 11, "Dec": 12
    }

    df["월_영문"] = df["일시"].str.split("-").str[0]
    df["연도_두자리"] = pd.to_numeric(
        df["일시"].str.split("-").str[1],
        errors="coerce"
    )

    df["월"] = df["월_영문"].map(month_dict)

    # 연도 변환
    df["연도"] = df["연도_두자리"].apply(
        lambda x: 2000 + int(x) if x < 100 else int(x)
    )

    # 기온 데이터 숫자로 변환
    df["최고기온(℃)"] = pd.to_numeric(
        df["최고기온(℃)"], errors="coerce"
    )

    df["최저기온(℃)"] = pd.to_numeric(
        df["최저기온(℃)"], errors="coerce"
    )

    df = df.dropna(
        subset=["연도", "월", "최고기온(℃)", "최저기온(℃)"]
    )

    return df


# 데이터 불러오기
try:
    df = load_data()

    # 월 선택
    month = st.selectbox(
        "📅 분석할 월을 선택하세요.",
        list(range(1, 13)),
        format_func=lambda x: f"{x}월"
    )

    # 선택한 월의 데이터 추출
    selected_df = df[df["월"] == month].copy()

    # 연도별 최고기온과 최저기온 정리
    result = selected_df.groupby("연도").agg(
        최고기온=("최고기온(℃)", "max"),
        최저기온=("최저기온(℃)", "min")
    ).reset_index()

    result = result.sort_values("연도")

    # 데이터가 있는지 확인
    if result.empty:
        st.warning("선택한 월의 데이터가 없습니다.")

    else:
        # 그래프 생성
        fig = go.Figure()

        # 최고기온 (빨간색)
        fig.add_trace(go.Scatter(
            x=result["연도"],
            y=result["최고기온"],
            mode="lines+markers",
            name="최고기온",
            line=dict(color="red", width=2),
            marker=dict(size=7)
        ))

        # 최저기온 (파란색)
        fig.add_trace(go.Scatter(
            x=result["연도"],
            y=result["최저기온"],
            mode="lines+markers",
            name="최저기온",
            line=dict(color="blue", width=2),
            marker=dict(size=7)
        ))

        # 그래프 레이아웃
        fig.update_layout(
            title="월별 기온분석",
            xaxis_title="연도",
            yaxis_title="온도 (℃)",
            legend_title="범례",
            hovermode="x unified",
            template="plotly_white",
            height=550
        )

        # 그래프 출력
        st.plotly_chart(fig, use_container_width=True)

        # 데이터 표 출력
        st.subheader(f"📊 {month}월 연도별 기온 데이터")
        st.dataframe(
            result.rename(columns={
                "연도": "연도",
                "최고기온": "최고기온(℃)",
                "최저기온": "최저기온(℃)"
            }),
            use_container_width=True,
            hide_index=True
        )

except FileNotFoundError:
    st.error("temperature.csv 파일을 찾을 수 없습니다. app.py와 같은 폴더에 넣어주세요.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")

