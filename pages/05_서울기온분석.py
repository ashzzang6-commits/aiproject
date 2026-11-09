
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
st.write("1907년부터 2026년까지의 연도별 최고기온과 최저기온을 비교해 보세요!")

# 데이터 불러오기
@st.cache_data
def load_data():

    df = pd.read_csv(
        "temperature.csv",
        encoding="cp949"
    )

    # 날짜에서 월과 연도 분리
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

    # 1907~1999년, 2000~2026년 구분
    year_num = pd.to_numeric(date_info[1], errors="coerce")

    df["연도"] = year_num.apply(
        lambda x: 1900 + int(x) if x >= 7
        else 2000 + int(x)
        if pd.notna(x) else None
    )

    # 기온 숫자 변환
    df["최고기온(℃)"] = pd.to_numeric(
        df["최고기온(℃)"], errors="coerce"
    )

    df["최저기온(℃)"] = pd.to_numeric(
        df["최저기온(℃)"], errors="coerce"
    )

    return df


# 데이터 실행
try:
    df = load_data()

    # 월 선택
    month = st.selectbox(
        "📅 분석할 월을 선택하세요.",
        range(1, 13),
        format_func=lambda x: f"{x}월"
    )

    # 선택한 월 데이터
    selected_df = df[df["월"] == month].copy()

    # 연도별 최고기온과 최저기온 계산
    result = selected_df.groupby("연도").agg(
        최고기온=("최고기온(℃)", "max"),
        최저기온=("최저기온(℃)", "min")
    ).reset_index()

    result = result.sort_values("연도")

    if result.empty:
        st.warning("선택한 월의 데이터가 없습니다.")

    else:
        # 그래프 생성
        fig = go.Figure()

        # 최고기온 그래프 (빨간색)
        fig.add_trace(go.Scatter(
            x=result["연도"],
            y=result["최고기온"],
            mode="lines+markers",
            name="최고기온",
            line=dict(color="red", width=2),
            marker=dict(size=5)
        ))

        # 최저기온 그래프 (파란색)
        fig.add_trace(go.Scatter(
            x=result["연도"],
            y=result["최저기온"],
            mode="lines+markers",
            name="최저기온",
            line=dict(color="blue", width=2),
            marker=dict(size=5)
        ))

        # 가로축 연도 설정
        tick_values = list(range(1907, 2027, 10))

        if 2026 not in tick_values:
            tick_values.append(2026)

        # 그래프 설정
        fig.update_layout(
            title="월별 기온분석",

            xaxis=dict(
                title="연도",
                range=[1907, 2026],
                tickmode="array",
                tickvals=tick_values,
                ticktext=[f"{y}년" for y in tick_values]
            ),

            yaxis=dict(
                title="온도 (℃)"
            ),

            legend=dict(
                title="범례",
                orientation="h",
                y=1.1,
                x=0.5,
                xanchor="center"
            ),

            hovermode="x unified",
            template="plotly_white",
            height=600
        )

        # 그래프 출력
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # 데이터 표
        st.subheader(f"📊 {month}월 연도별 기온 데이터")

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

except FileNotFoundError:
    st.error("temperature.csv 파일을 찾을 수 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
