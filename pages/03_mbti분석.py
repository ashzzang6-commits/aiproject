import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 페이지 기본 설정
st.set_page_config(
    page_title="국가별 MBTI 비율 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 국가별 MBTI 유형 비율 분석")
st.write("국가를 선택하면 해당 국가의 16가지 MBTI 비율을 막대그래프로 확인하실 수 있습니다.")

# 1. 데이터 로드 (UTF-8 및 CP949 자동 대응)
@st.cache_data
def load_data():
    file_path = 'countriesMBTI_16types.csv'
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except (UnicodeDecodeError, FileNotFoundError):
        df = pd.read_csv(file_path, encoding='cp949')
    return df

try:
    df = load_data()
    mbti_cols = [c for c in df.columns if c != 'Country']
    
    # 2. 국가 선택 사이드바
    country_list = sorted(df['Country'].unique())
    default_idx = country_list.index("South Korea") if "South Korea" in country_list else 0
    selected_country = st.sidebar.selectbox("국가를 선택하세요", country_list, index=default_idx)
    
    # 선택 국가 데이터 추출 및 퍼센트 환산
    country_data = df[df['Country'] == selected_country].iloc[0]
    mbti_series = country_data[mbti_cols].astype(float) * 100
    mbti_sorted = mbti_series.sort_values(ascending=False)
    
    # 3. 색상 팔레트 생성
    # 1등: 빨간색 (#E63946)
    # 2등 ~ 16등: 진한 파랑 (#1D3557)에서 연한 파랑 (#A8DADC) 그라데이션
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_grad", ["#1D3557", "#A8DADC"])
    num_other_types = len(mbti_sorted) - 1
    
    colors = ["#E63946"]
    for i in range(num_other_types):
        colors.append(mcolors.to_hex(cmap(i / max(1, num_other_types - 1))))

    # 4. 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(mbti_sorted.index, mbti_sorted.values, color=colors, edgecolor='none')
    
    # 수치 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title(f"[{selected_country}] MBTI Distribution", fontsize=16, pad=15, fontweight='bold')
    ax.set_xlabel("MBTI Types", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_ylim(0, max(mbti_sorted.values) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # Streamlit 출력
    st.pyplot(fig)
    
    # 상위 5개 요약 테이블
    st.subheader(f"📌 {selected_country} 상위 MBTI Top 5")
    top5_df = pd.DataFrame({
        'MBTI 유형': mbti_sorted.head(5).index,
        '비율 (%)': [f"{val:.2f}%" for val in mbti_sorted.head(5).values]
    }).reset_index(drop=True)
    st.table(top5_df)

except FileNotFoundError:
    st.error("⚠️ 'countriesMBTI_16types.csv' 파일을 찾을 수 없습니다. github 리포지토리 최상위 경로에 파일을 업로드해 주세요.")
except Exception as e:
    st.error(f"⚠️ 오류가 발생했습니다: {e}")
