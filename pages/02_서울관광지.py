
import streamlit as st
import folium
from streamlit_folium import st_folium

# 페이지 설정
st.set_page_config(
    page_title="서울 관광지 TOP 10",
    page_icon="🗺️",
    layout="wide"
)

# 제목
st.title("🗺️ 서울 인기 관광지 TOP 10")
st.subheader("외국인 관광객에게 사랑받는 서울의 대표 명소!")
st.write("지도에서 숫자 마커를 클릭하면 관광지 정보를 확인할 수 있어요.")

# 관광지 데이터
places = [
    {
        "name": "경복궁",
        "eng": "Gyeongbokgung Palace",
        "lat": 37.5796,
        "lon": 126.9770,
        "desc": "조선 왕조의 대표 궁궐입니다. 아름다운 전통 건축물과 한복 체험을 즐길 수 있습니다."
    },
    {
        "name": "명동",
        "eng": "Myeongdong",
        "lat": 37.5636,
        "lon": 126.9869,
        "desc": "쇼핑과 K-뷰티, 다양한 길거리 음식으로 유명한 서울의 대표 번화가입니다."
    },
    {
        "name": "N서울타워",
        "eng": "N Seoul Tower",
        "lat": 37.5512,
        "lon": 126.9882,
        "desc": "남산에 위치한 전망대로 서울의 아름다운 야경을 감상할 수 있습니다."
    },
    {
        "name": "북촌한옥마을",
        "eng": "Bukchon Hanok Village",
        "lat": 37.5826,
        "lon": 126.9830,
        "desc": "전통 한옥이 모여 있는 곳으로 한국의 전통적인 분위기를 느낄 수 있습니다."
    },
    {
        "name": "인사동",
        "eng": "Insadong",
        "lat": 37.5710,
        "lon": 126.9850,
        "desc": "한국 전통 공예품과 기념품, 전통 찻집을 만날 수 있는 문화 거리입니다."
    },
    {
        "name": "창덕궁",
        "eng": "Changdeokgung Palace",
        "lat": 37.5794,
        "lon": 126.9910,
        "desc": "유네스코 세계문화유산으로 아름다운 궁궐과 후원이 유명합니다."
    },
    {
        "name": "홍대",
        "eng": "Hongdae",
        "lat": 37.5563,
        "lon": 126.9236,
        "desc": "거리 공연과 개성 있는 카페, 쇼핑을 즐길 수 있는 젊음의 거리입니다."
    },
    {
        "name": "광장시장",
        "eng": "Gwangjang Market",
        "lat": 37.5700,
        "lon": 126.9998,
        "desc": "빈대떡과 마약김밥 등 다양한 한국 전통 음식을 맛볼 수 있습니다."
    },
    {
        "name": "동대문디자인플라자",
        "eng": "Dongdaemun Design Plaza",
        "lat": 37.5671,
        "lon": 127.0095,
        "desc": "독특하고 미래적인 건축 디자인으로 유명한 전시 및 문화 공간입니다."
    },
    {
        "name": "롯데월드",
        "eng": "Lotte World",
        "lat": 37.5111,
        "lon": 127.0982,
        "desc": "다양한 놀이기구와 퍼레이드를 즐길 수 있는 대형 테마파크입니다."
    }
]

# 서울 지도 생성
seoul_map = folium.Map(
    location=[37.5665, 126.9780],
    zoom_start=11,
    tiles="OpenStreetMap"
)

# 지도에 관광지 표시
for i, place in enumerate(places, start=1):

    popup_text = f"""
    <div style="width:220px">
        <h4>{i}. {place['name']}</h4>
        <p>{place['eng']}</p>
        <p>{place['desc']}</p>
    </div>
    """

    folium.Marker(
        location=[place["lat"], place["lon"]],
        tooltip=f"{i}. {place['name']}",
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.DivIcon(
            html=f"""
            <div style="
                background-color:#2878D0;
                color:white;
                border:2px solid white;
                border-radius:50%;
                width:30px;
                height:30px;
                text-align:center;
                line-height:27px;
                font-weight:bold;
                font-size:14px;
                box-shadow:0 2px 5px #777;">
                {i}
            </div>
            """
        )
    ).add_to(seoul_map)

# 지도 출력
st_folium(
    seoul_map,
    width=None,
    height=550,
    use_container_width=True,
    returned_objects=[]
)

# 관광지 소개
st.divider()
st.header("📍 서울 관광지 TOP 10 소개")

for i, place in enumerate(places, start=1):

    with st.container(border=True):
        st.subheader(f"{i}. {place['name']}")
        st.caption(place["eng"])
        st.write(place["desc"])

# 안내 문구
st.info(
    "이 목록은 외국인 관광객에게 잘 알려진 서울의 대표 관광지를 "
    "소개하기 위한 선정 목록이며, 공식 방문객 순위는 아닙니다."
)

st.caption("🗺️ Map data © OpenStreetMap contributors")
