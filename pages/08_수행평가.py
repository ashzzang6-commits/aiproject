
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from urllib.parse import quote_plus

# -----------------------------
# 1. 기본 설정
# -----------------------------
st.set_page_config(
    page_title="Mood Music",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #100d20, #21143b);
    color: white;
}
h1, h2, h3 {
    color: #e4c6ff !important;
}
div[data-testid="stMetric"] {
    background-color: #30204b;
    padding: 15px;
    border-radius: 12px;
}
.stButton > button {
    background-color: #9b59d0;
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #bd83ec;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# 2. 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent

    candidates = [
        base / "songs.zip",
        base.parent / "songs.zip"
    ]

    csv_candidates = [
        base / "spotify_songs.csv",
        base.parent / "spotify_songs.csv"
    ]

    for path in candidates:
        if path.exists():
            with zipfile.ZipFile(path) as z:
                with z.open("spotify_songs.csv") as f:
                    return pd.read_csv(f)

    for path in csv_candidates:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(
        "songs.zip 또는 spotify_songs.csv 파일을 찾을 수 없습니다."
    )


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# -----------------------------
# 3. 데이터 정리
# -----------------------------
required = [
    "track_name", "track_artist",
    "playlist_genre", "track_popularity",
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness"
]

df = df.dropna(subset=required).copy()

df["playlist_genre"] = df["playlist_genre"].astype(str)
df["track_name"] = df["track_name"].astype(str)
df["track_artist"] = df["track_artist"].astype(str)

features = [
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "instrumentalness"
]


# -----------------------------
# 4. 제목
# -----------------------------
st.title("🎧 MOOD MUSIC")
st.subheader("오늘의 기분에 어울리는 음악을 찾아보세요.")
st.write("당신의 취향과 분위기를 분석해 노래를 추천해 드립니다.")

st.divider()

# 데이터 정보
c1, c2, c3 = st.columns(3)

c1.metric("전체 수록곡", f"{len(df):,}곡")
c2.metric("음악 장르", f"{df['playlist_genre'].nunique()}개")
c3.metric("아티스트", f"{df['track_artist'].nunique():,}명")

st.divider()


# -----------------------------
# 5. 사용자 취향 설정
# -----------------------------
st.header("💜 나의 음악 취향 설정")

col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox(
        "지금 어떤 기분인가요?",
        ["신나는", "행복한", "차분한", "우울한",
         "감성적인", "집중하고 싶은"]
    )

    genres = ["전체"] + sorted(
        df["playlist_genre"].unique().tolist()
    )

    genre = st.selectbox(
        "원하는 음악 장르",
        genres
    )

with col2:
    min_popularity = st.slider(
        "최소 음악 인기도",
        0, 100, 30
    )

    count = st.slider(
        "추천받을 노래 수",
        5, 20, 10
    )


# -----------------------------
# 6. 기분별 음악 특성
# -----------------------------
mood_features = {
    "신나는": [0.90, 0.85, 0.85, 0.10, 0.05],
    "행복한": [0.65, 0.70, 0.90, 0.25, 0.05],
    "차분한": [0.25, 0.35, 0.55, 0.75, 0.20],
    "우울한": [0.25, 0.40, 0.15, 0.65, 0.10],
    "감성적인": [0.40, 0.50, 0.40, 0.60, 0.05],
    "집중하고 싶은": [0.25, 0.30, 0.40, 0.50, 0.75]
}

target = np.array(mood_features[mood])


# -----------------------------
# 7. 노래 추천 알고리즘
# -----------------------------
def recommend_songs(data, mood, genre, popularity, count):

    result = data.copy()

    # 장르 필터
    if genre != "전체":
        result = result[
            result["playlist_genre"] == genre
        ]

    # 인기도 필터
    result = result[
        result["track_popularity"] >= popularity
    ]

    if result.empty:
        return result

    # 음악 특성 비교
    values = result[features].to_numpy(dtype=float)

    # 각 음악과 목표 분위기의 차이 계산
    weights = np.array([1.2, 1.0, 1.2, 0.8, 0.8])

    distances = np.sqrt(
        np.sum(
            ((values - target) ** 2) * weights,
            axis=1
        )
    )

    result["similarity"] = 1 / (1 + distances)

    # 유사도가 높은 순서로 정렬
    result = result.sort_values(
        by=["similarity", "track_popularity"],
        ascending=False
    )

    # 같은 가수의 노래가 반복되는 것 방지
    result = result.drop_duplicates(
        subset=["track_name", "track_artist"]
    )

    return result.head(count)


# -----------------------------
# 8. 추천 실행
# -----------------------------
st.divider()

st.header("✨ 나에게 맞는 음악 추천")

if st.button("🎵 노래 추천받기", use_container_width=True):

    result = recommend_songs(
        df,
        mood,
        genre,
        min_popularity,
        count
    )

    if result.empty:
        st.warning("조건에 맞는 노래가 없습니다. 필터를 변경해 주세요.")

    else:
        st.session_state["recommendations"] = result


# -----------------------------
# 9. 추천 결과
# -----------------------------
if "recommendations" in st.session_state:

    result = st.session_state["recommendations"]

    st.success(f"총 {len(result)}곡을 추천합니다!")

    for i, row in enumerate(
        result.to_dict("records"), start=1
    ):

        with st.container(border=True):

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown(f"## 🎧 {i:02d}")

            with col2:
                st.subheader(row["track_name"])

                st.write(f"🎤 아티스트: {row['track_artist']}")

                st.write(
                    f"🎼 장르: {row['playlist_genre']}"
                )

                st.write(
                    f"⭐ 인기도: {row['track_popularity']}/100"
                )

                st.write(
                    f"💜 분위기 일치도: "
                    f"{row['similarity'] * 100:.1f}%"
                )

                search_url = (
                    "https://open.spotify.com/search/"
                    + quote_plus(
                        row["track_name"] + " " + row["track_artist"]
                    )
                )

                st.markdown(
                    f"[Spotify에서 노래 검색하기]({search_url})"
                )

    # -------------------------
    # 10. 추천 음악 분석 그래프
    # -------------------------
    st.divider()
    st.header("📊 추천 음악 분석")

    chart_data = result[
        ["track_name", "energy", "danceability", "valence"]
    ].set_index("track_name")

    st.write("추천된 음악의 에너지, 춤추기 좋은 정도, 밝은 분위기")

    st.bar_chart(chart_data)

    # -------------------------
    # 11. 데이터 다운로드
    # -------------------------
    st.divider()

    csv = result.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 추천 노래 목록 다운로드",
        data=csv,
        file_name="my_music_recommendations.csv",
        mime="text/csv",
        use_container_width=True
    )


# -----------------------------
# 12. 전체 데이터 탐색
# -----------------------------
with st.expander("🔎 전체 음악 데이터 살펴보기"):

    search = st.text_input(
        "노래 제목 또는 가수 검색"
    )

    if search:
        searched = df[
            df["track_name"].str.contains(
                search, case=False, na=False
            ) |
            df["track_artist"].str.contains(
                search, case=False, na=False
            )
        ]

        st.dataframe(
            searched[
                ["track_name", "track_artist",
                 "playlist_genre", "track_popularity"]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.dataframe(
            df[
                ["track_name", "track_artist",
                 "playlist_genre", "track_popularity"]
            ].head(100),
            use_container_width=True,
            hide_index=True
        )


st.divider()

st.caption("MOOD MUSIC | Find your own sound 🎧")
