
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
import re

# -------------------------
# 기본 설정
# -------------------------
st.set_page_config(
    page_title="Mood Music",
    page_icon="🎧",
    layout="wide"
)

# -------------------------
# 디자인
# -------------------------
st.markdown("""
<style>
.stApp {
    background-color: #211638;
    color: #FFFFFF;
}

h1, h2, h3, p, label, span, div {
    color: #FFFFFF;
}

h1 {
    color: #D9B8FF !important;
}

.stSelectbox label,
.stSlider label {
    color: #FFFFFF !important;
}

div[data-baseweb="select"] {
    background-color: #FFFFFF;
    border-radius: 8px;
}

.stButton > button {
    background-color: #A66BEA;
    color: white;
    border: none;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #C18BFF;
    color: white;
}

div[data-testid="stMetric"] {
    background-color: #382653;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stMetricValue"] {
    color: #D9B8FF;
}

div[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
}

hr {
    border-color: #76569B;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# 데이터 불러오기
# -------------------------
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent

    paths = [
        base / "songs.zip",
        base.parent / "songs.zip"
    ]

    for path in paths:
        if path.exists():
            with zipfile.ZipFile(path) as z:
                with z.open("spotify_songs.csv") as f:
                    return pd.read_csv(f)

    raise FileNotFoundError("songs.zip 파일을 찾을 수 없습니다.")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 오류: {e}")
    st.stop()


# -------------------------
# 데이터 정리
# -------------------------
features = [
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "instrumentalness"
]

required = [
    "track_name",
    "track_artist",
    "playlist_genre",
    "track_popularity"
] + features

df = df.dropna(subset=required).copy()


# -------------------------
# K-POP 검색용 분류
# -------------------------
kpop_artists = [
    "BTS", "BLACKPINK", "TWICE", "EXO",
    "SEVENTEEN", "NCT", "STRAY KIDS",
    "IVE", "AESPA", "NEWJEANS",
    "LE SSERAFIM", "ITZY", "TXT",
    "RED VELVET", "GOT7", "BIGBANG",
    "G-DRAGON", "JUNGKOOK", "JIMIN",
    "PSY", "IU", "TAEYEON", "ENHYPEN"
]

def is_kpop(row):
    artist = str(row["track_artist"]).upper()
    name = str(row["track_name"])

    has_korean = bool(
        re.search(r"[가-힣]", artist + name)
    )

    known_artist = any(
        name in artist for name in kpop_artists
    )

    return has_korean or known_artist


# -------------------------
# 음악 추천 설정
# -------------------------
moods = {
    "신나는": [0.90, 0.85, 0.85, 0.10, 0.05],
    "행복한": [0.65, 0.70, 0.90, 0.25, 0.05],
    "차분한": [0.25, 0.35, 0.55, 0.75, 0.20],
    "감성적인": [0.40, 0.50, 0.40, 0.60, 0.05],
    "집중하고 싶은": [0.25, 0.30, 0.40, 0.50, 0.75]
}


def recommend(data, mood, genre, count):

    result = data.copy()

    if genre == "K-POP":
        result = result[result.apply(is_kpop, axis=1)]

    elif genre != "전체":
        result = result[
            result["playlist_genre"] == genre.lower()
        ]

    if result.empty:
        return result

    target = np.array(moods[mood])

    values = result[features].to_numpy()

    distance = np.sqrt(
        np.sum((values - target) ** 2, axis=1)
    )

    result["similarity"] = 1 / (1 + distance)

    result = result.sort_values(
        "similarity", ascending=False
    )

    result = result.drop_duplicates(
        subset=["track_name", "track_artist"]
    )

    return result.head(count)


# -------------------------
# 메인 화면
# -------------------------
st.title("🎧 MOOD MUSIC")

st.write("오늘의 기분에 어울리는 음악을 찾아보세요.")

st.divider()

# 데이터 정보
col1, col2, col3 = st.columns(3)

col1.metric("전체 노래", f"{len(df):,}곡")
col2.metric("음악 장르", "7개")
col3.metric("아티스트", f"{df['track_artist'].nunique():,}명")

st.divider()


# -------------------------
# 사용자 선택
# -------------------------
st.subheader("💜 나의 음악 취향")

col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox(
        "오늘의 기분",
        list(moods.keys())
    )

with col2:
    genre = st.selectbox(
        "음악 장르",
        [
            "전체",
            "K-POP",
            "POP",
            "EDM",
            "RAP",
            "R&B",
            "ROCK",
            "LATIN"
        ]
    )

count = st.slider(
    "추천받을 노래 수",
    5, 20, 10
)

st.write("")

if st.button("🎵 노래 추천받기", use_container_width=True):

    result = recommend(df, mood, genre, count)

    if result.empty:
        st.session_state.pop("songs", None)
        st.warning(
            "해당 장르의 노래가 데이터에 없습니다. "
            "다른 장르를 선택해 주세요."
        )
    else:
        st.session_state["songs"] = result


# -------------------------
# 추천 결과
# -------------------------
if "songs" in st.session_state:

    result = st.session_state["songs"]

    st.divider()
    st.subheader("✨ 추천 음악")

    st.write(f"총 {len(result)}곡을 추천합니다.")

    for i, row in enumerate(
        result.to_dict("records"), 1
    ):

        with st.container(border=True):

            st.markdown(
                f"### 🎧 {i}. {row['track_name']}"
            )

            st.write(f"아티스트: {row['track_artist']}")

            st.write(
                f"장르: {row['playlist_genre'].upper()}"
            )

            st.write(
                f"추천 일치도: {row['similarity'] * 100:.1f}%"
            )

            from urllib.parse import quote_plus

            url = (
                "https://open.spotify.com/search/"
                + quote_plus(
                    row["track_name"] + " " + row["track_artist"]
                )
            )

            st.markdown(
                f"[Spotify에서 노래 듣기 및 검색하기]({url})"
            )

    # 다운로드
    st.divider()

    csv = result.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 추천 노래 저장하기",
        csv,
        "mood_music.csv",
        "text/csv",
        use_container_width=True
    )


# -------------------------
# 하단
# -------------------------
st.divider()

st.caption("MOOD MUSIC | Your music, your mood.")
