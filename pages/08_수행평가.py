
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from urllib.parse import quote_plus
import re

# ==================================
# 1. 기본 설정
# ==================================

st.set_page_config(
    page_title="MOOD MUSIC",
    page_icon="🎧",
    layout="wide"
)

# ==================================
# 2. 디자인
# ==================================

st.markdown("""
<style>
.stApp {
    background-color: #211638;
    color: white;
}

h1 {
    color: #D9B8FF !important;
}

h2, h3, p, label, span {
    color: white !important;
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
    color: #D9B8FF !important;
}

div[data-testid="stDataFrame"] {
    background-color: white;
}

hr {
    border-color: #76569B;
}
</style>
""", unsafe_allow_html=True)


# ==================================
# 3. 데이터 불러오기
# ==================================

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

    raise FileNotFoundError(
        "songs.zip 파일을 찾을 수 없습니다."
    )


try:
    df = load_data()

except Exception as e:

    st.error(f"데이터 오류: {e}")
    st.stop()


# ==================================
# 4. 데이터 정리
# ==================================

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
    df["playlist_genre"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["track_artist"] = (
    df["track_artist"].astype(str)
)

df["track_name"] = (
    df["track_name"].astype(str)
)


# ==================================
# 5. K-POP 아티스트 한글 변환
# ==================================

kpop_artists = {
    "BTS": "방탄소년단",
    "BLACKPINK": "블랙핑크",
    "TWICE": "트와이스",
    "EXO": "엑소",
    "SEVENTEEN": "세븐틴",
    "NCT": "엔시티",
    "STRAY KIDS": "스트레이 키즈",
    "IVE": "아이브",
    "AESPA": "에스파",
    "NEWJEANS": "뉴진스",
    "LE SSERAFIM": "르세라핌",
    "ITZY": "있지",
    "TXT": "투모로우바이투게더",
    "RED VELVET": "레드벨벳",
    "GOT7": "갓세븐",
    "BIGBANG": "빅뱅",
    "G-DRAGON": "지드래곤",
    "JUNGKOOK": "정국",
    "JIMIN": "지민",
    "PSY": "싸이",
    "IU": "아이유",
    "TAEYEON": "태연",
    "ENHYPEN": "엔하이픈",
    "ATEEZ": "에이티즈",
    "MONSTA X": "몬스타엑스",
    "MAMAMOO": "마마무",
    "SUPER JUNIOR": "슈퍼주니어",
    "SHINEE": "샤이니",
    "2NE1": "투애니원",
    "WONDER GIRLS": "원더걸스",
    "GIRLS GENERATION": "소녀시대",
    "SNSD": "소녀시대"
}


def korean_artist_name(artist):

    artist_upper = str(artist).upper().strip()

    for eng, kor in kpop_artists.items():

        if eng in artist_upper:
            return kor

    # 등록되지 않은 한글 가수는 원래 이름 유지
    return artist


def is_kpop(artist):

    artist_upper = str(artist).upper().strip()

    # 등록된 K-POP 가수인지 확인
    for name in kpop_artists.keys():

        if name in artist_upper:
            return True

    # 한글 이름을 사용하는 가수 확인
    if re.search(r"[가-힣]", str(artist)):
        return True

    return False


# ==================================
# 6. 기분별 음악 특성
# ==================================

moods = {

    "신나는": [
        0.90, 0.85, 0.85, 0.10, 0.05
    ],

    "행복한": [
        0.65, 0.70, 0.90, 0.25, 0.05
    ],

    "차분한": [
        0.25, 0.35, 0.55, 0.75, 0.20
    ],

    "감성적인": [
        0.40, 0.50, 0.40, 0.60, 0.05
    ],

    "집중하고 싶은": [
        0.25, 0.30, 0.40, 0.50, 0.75
    ]

}


# ==================================
# 7. 노래 추천 함수
# ==================================

def recommend(data, mood, genre, count):

    result = data.copy()

    # 장르 필터링

    genre_mapping = {

        "POP": "pop",
        "EDM": "edm",
        "RAP": "rap",
        "R&B": "r&b",
        "ROCK": "rock",
        "LATIN": "latin"

    }

    if genre == "K-POP":

        result = result[
            result["track_artist"].apply(is_kpop)
        ].copy()

    elif genre != "전체":

        selected_genre = genre_mapping[genre]

        result = result[
            result["playlist_genre"] == selected_genre
        ].copy()

    # 데이터가 없는 경우

    if result.empty:
        return result

    # 기분과 음악 특성 비교

    target = np.array(moods[mood])

    values = result[features].to_numpy(dtype=float)

    distance = np.sqrt(
        np.sum((values - target) ** 2, axis=1)
    )

    result["similarity"] = 1 / (1 + distance)

    # 추천 순서 정렬

    result = result.sort_values(
        by="similarity",
        ascending=False
    )

    # 중복 제거

    result = result.drop_duplicates(
        subset=["track_name", "track_artist"]
    )

    return result.head(count)


# ==================================
# 8. 메인 화면
# ==================================

st.title("🎧 MOOD MUSIC")

st.write(
    "오늘의 기분에 어울리는 음악을 찾아보세요."
)

st.divider()


# ==================================
# 9. 데이터 정보
# ==================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 노래",
    f"{len(df):,}곡"
)

col2.metric(
    "음악 장르",
    "7개"
)

col3.metric(
    "아티스트",
    f"{df['track_artist'].nunique():,}명"
)

st.divider()


# ==================================
# 10. 음악 취향 선택
# ==================================

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
    5,
    20,
    10
)


# ==================================
# 11. 추천 실행
# ==================================

if st.button(
    "🎵 노래 추천받기",
    use_container_width=True
):

    st.session_state.pop("songs", None)

    result = recommend(
        df,
        mood,
        genre,
        count
    )

    if result.empty:

        st.warning(
            "선택한 장르에 해당하는 노래가 없습니다."
        )

    else:

        st.session_state["songs"] = result
        st.session_state["selected_genre"] = genre


# ==================================
# 12. 추천 결과
# ==================================

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

    # 추천 노래 표시

    for i, row in enumerate(
        result.to_dict("records"),
        1
    ):

        with st.container(border=True):

            st.markdown(
                f"### 🎧 {i}. {row['track_name']}"
            )

            # K-POP 가수 한글 표시

            if selected_genre == "K-POP":

                artist = korean_artist_name(
                    row["track_artist"]
                )

            else:

                artist = row["track_artist"]

            st.write(
                f"🎤 아티스트: {artist}"
            )

            if selected_genre == "K-POP":

                st.write("🎼 장르: K-POP")

            elif selected_genre == "전체":

                st.write(
                    f"🎼 장르: {row['playlist_genre'].upper()}"
                )

            else:

                st.write(
                    f"🎼 장르: {selected_genre}"
                )

            st.write(
                f"💜 분위기 일치도: "
                f"{row['similarity'] * 100:.1f}%"
            )

            # Spotify 검색

            url = (
                "https://open.spotify.com/search/"
                + quote_plus(
                    row["track_name"]
                    + " "
                    + row["track_artist"]
                )
            )

            st.markdown(
                f"[Spotify에서 노래 검색하기]({url})"
            )


    # ==================================
    # 13. 추천 목록 다운로드
    # ==================================

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


# ==================================
# 14. 하단
# ==================================

st.divider()

st.caption(
    "MOOD MUSIC | Your music, your mood 🎧"
)
