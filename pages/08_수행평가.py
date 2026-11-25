
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
import re
from urllib.parse import quote_plus

# -------------------------
# 1. 기본 설정
# -------------------------
st.set_page_config(
    page_title="Mood Music",
    page_icon="🎧",
    layout="wide"
)

# -------------------------
# 2. 디자인
# -------------------------
st.markdown("""
<style>
.stApp {
    background-color: #211638;
    color: white;
}

h1, h2, h3, p, label, span {
    color: white !important;
}

h1 {
    color: #D9B8FF !important;
}

.stSelectbox label,
.stSlider label {
    color: white !important;
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

</style>
""", unsafe_allow_html=True)


# -------------------------
# 3. 데이터 불러오기
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
# 4. 데이터 정리
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

df["playlist_genre"] = (
    df["playlist_genre"].astype(str).str.strip().str.lower()
)

df["track_artist"] = df["track_artist"].astype(str)
df["track_name"] = df["track_name"].astype(str)


# -------------------------
# 5. K-POP 가수 분류
# -------------------------
kpop_artists = [
    "BTS", "BLACKPINK", "TWICE", "EXO",
    "SEVENTEEN", "NCT", "STRAY KIDS",
    "IVE", "AESPA", "NEWJEANS",
    "LE SSERAFIM", "ITZY", "TXT",
    "RED VELVET", "GOT7", "BIGBANG",
    "G-DRAGON", "JUNGKOOK", "JIMIN",
    "PSY", "IU", "TAEYEON", "ENHYPEN",
    "ATEEZ", "MONSTA X", "MAMAMOO",
    "SUPER JUNIOR", "SHINee", "2NE1",
    "WONDER GIRLS", "GIRLS GENERATION",
    "SNSD", "TOMORROW X TOGETHER"
]

def is_kpop(artist):
    artist = artist.upper().strip()

    # 등록된 K-POP 가수 확인
    for name in kpop_artists:
        if name in artist:
            return True

    # 한글 이름을 사용하는 가수 확인
    if re.search(r"[가-힣]", artist):
        return True

    return False


# -------------------------
# 6. 기분별 음악 특성
# -------------------------
moods = {
    "신나는": [0.90, 0.85, 0.85, 0.10, 0.05],
    "행복한": [0.65, 0.70, 0.90, 0.25, 0.05],
    "차분한": [0.25, 0.35, 0.55, 0.75, 0.20],
    "감성적인": [0.40, 0.50, 0.40, 0.60, 0.05],
    "집중하고 싶은": [0.25, 0.30, 0.40, 0.50, 0.75]
}


# -------------------------
# 7. 노래 추천 함수
# -------------------------
def recommend(data, mood, genre, count):

    result = data.copy()

    # 장르를 먼저 정확하게 필터링
    if genre == "K-POP":

        result = result[
            result["track_artist"].apply(is_kpop)
        ].copy()

    else:

        genre_mapping = {
            "POP": "pop",
            "EDM": "edm",
            "RAP": "rap",
            "R&B": "r&b",
            "ROCK": "rock",
            "LATIN": "latin"
        }

        selected_genre = genre_mapping.get(genre)

        if selected_genre:
            result = result[
                result["playlist_genre"] == selected_genre
            ].copy()

    # 데이터가 없는 경우
    if result.empty:
        return result

    # 선택한 기분과 음악 특성 비교
    target = np.array(moods[mood])

    values = result[features].to_numpy(dtype=float)

    distance = np.sqrt(
        np.sum((values - target) ** 2, axis=1)
    )

    result["similarity"] = 1 / (1 + distance)

    # 유사도가 높은 노래부터 정렬
    result = result.sort_values(
        by="similarity",
        ascending=False
    )

    # 중복 노래 제거
    result = result.drop_duplicates(
        subset=["track_name", "track_artist"]
    )

    # 선택한 장르의 노래만 반환하는지 마지막으로 확인
    if genre != "K-POP":
        result = result[
            result["playlist_genre"] == selected_genre
        ]

    else:
        result = result[
            result["track_artist"].apply(is_kpop)
        ]

    return result.head(count)


# -------------------------
# 8. 메인 화면
# -------------------------
st.title("🎧 MOOD MUSIC")
st.write("오늘의 기분에 어울리는 음악을 찾아보세요.")

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric("전체 노래", f"{len(df):,}곡")
col2.metric("음악 장르", "7개")
col3.metric("아티스트", f"{df['track_artist'].nunique():,}명")

st.divider()


# -------------------------
# 9. 음악 취향 선택
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
        "원하는 음악 장르",
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


# -------------------------
# 10. 추천 버튼
# -------------------------
if st.button("🎵 노래 추천받기", use_container_width=True):

    # 이전 결과 삭제
    st.session_state.pop("songs", None)

    result = recommend(
        df,
        mood,
        genre,
        count
    )

    if result.empty:
        st.warning(
            "선택한 장르에 해당하는 노래가 없습니다. "
            "다른 장르를 선택해 주세요."
        )
    else:
        st.session_state["songs"] = result
        st.session_state["selected_genre"] = genre


# -------------------------
# 11. 추천 결과
# -------------------------
if "songs" in st.session_state:

    result = st.session_state["songs"]
    selected_genre = st.session_state["selected_genre"]

    st.divider()
    st.subheader("✨ 추천 음악")

    st.write(
        f"선택한 장르: {selected_genre}"
    )

    st.write(
        f"총 {len(result)}곡을 추천합니다."
    )

    # 장르 일치 여부 표시
    if selected_genre != "K-POP":
        genre_mapping = {
            "POP": "pop",
            "EDM": "edm",
            "RAP": "rap",
            "R&B": "r&b",
            "ROCK": "rock",
            "LATIN": "latin"
        }

        expected = genre_mapping.get(selected_genre)

        if selected_genre != "전체":
            assert (result["playlist_genre"] == expected).all()

    for i, row in enumerate(
        result.to_dict("records"), 1
    ):

        with st.container(border=True):

            st.markdown(
                f"### 🎧 {i}. {row['track_name']}"
            )

            st.write(
                f"🎤 아티스트: {row['track_artist']}"
            )

            if selected_genre == "K-POP":
                st.write("🎼 장르: K-POP (가수 기준 추정)")
            else:
                st.write(f"🎼 장르: {selected_genre}")

            st.write(
                f"💜 분위기 일치도: "
                f"{row['similarity'] * 100:.1f}%"
            )

            url = (
                "https://open.spotify.com/search/"
                + quote_plus(
                    row["track_name"] + " " + row["track_artist"]
                )
            )

            st.markdown(
                f"[Spotify에서 노래 검색하기]({url})"
            )

    # -------------------------
    # 12. 추천 목록 다운로드
    # -------------------------
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
# 13. 하단
# -------------------------
st.divider()
st.caption("MOOD MUSIC | Your music, your mood 🎧")
