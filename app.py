# app.py
import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="EuroSAT Классификация", page_icon="🛰️", layout="wide")

BACKEND_URL = "http://localhost:8080"

# Session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'channels_count' not in st.session_state:
    st.session_state.channels_count = 3
if 'preview_image' not in st.session_state:
    st.session_state.preview_image = None
if 'selected_channels' not in st.session_state:
    st.session_state.selected_channels = None

st.title("🛰️ EuroSAT Классификация")

# CSS стили
st.markdown("""
<style>
    .stButton > button { width: 100%; }
    .legend-item { display: flex; align-items: center; margin: 0.3rem 0; font-size: 0.85rem; }
    .color-box { width: 16px; height: 16px; border-radius: 2px; margin-right: 8px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# Компактная легенда в sidebar
with st.sidebar:
    st.header("🎨 Легенда")
    st.markdown("""
    <div style="margin-top: 0.5rem;">
        <div class="legend-item">
            <span class="color-box" style="background: rgb(255,255,0);"></span>
            <span>🌾 Однолетние культуры</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(34,139,34);"></span>
            <span>🌲 Лес</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(124,252,0);"></span>
            <span>🌿 Травы</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(128,128,128);"></span>
            <span>🛣️ Дороги</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(139,69,19);"></span>
            <span>🏭 Промзоны</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(152,251,152);"></span>
            <span>🌱 Пастбища</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(255,140,0);"></span>
            <span>🍇 Многолетние культуры</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(255,0,0);"></span>
            <span>🏘️ Жилые районы</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(0,0,255);"></span>
            <span>🏞️ Реки</span>
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: rgb(0,191,255);"></span>
            <span>🌊 Водоёмы</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <small>
    <b>О приложении</b><br>
    Классификация спутниковых снимков с помощью нейросети EuroSAT<br><br>
    🚀 TensorFlow & FastAPI
    </small>
    """, unsafe_allow_html=True)

# Загрузка файла
uploaded_file = st.file_uploader(
    "Загрузите растровое изображение", 
    type=["tif", "tiff", "jpg", "png", "jpeg"],
    help="Поддерживаются: GeoTIFF, TIFF, JPG, PNG"
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    
    # Проверка каналов
    with st.spinner("Анализ..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/check_channels/", files={"file": (uploaded_file.name, file_bytes)})
            if resp.status_code == 200:
                st.session_state.channels_count = resp.json()["channels"]
        except:
            st.session_state.channels_count = 3
    
    # Выбор каналов для мультиспектральных
    channels = None
    if st.session_state.channels_count > 3:
        st.info(f"🔍 Мультиспектральный снимок ({st.session_state.channels_count} каналов)")
        
        with st.expander("Настройка каналов", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                cols = st.columns(3)
                with cols[0]:
                    r = st.number_input('R', 1, st.session_state.channels_count, 4)
                with cols[1]:
                    g = st.number_input('G', 1, st.session_state.channels_count, 3)
                with cols[2]:
                    b = st.number_input('B', 1, st.session_state.channels_count, 2)
                channels = f"{r},{g},{b}"
            
            with col2:
                preset = st.selectbox("Пресет", ["Custom", "4,3,2", "8,4,3"])
                if preset != "Custom":
                    st.info(f"Применить: {preset}")
            
            # Обновление превью
            if st.button("🔄 Обновить превью"):
                st.session_state.selected_channels = channels
                with st.spinner("Создание превью..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/get_preview/",
                            files={"file": (uploaded_file.name, file_bytes)},
                            params={"channels": channels}
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            preview_bytes = base64.b64decode(data['preview'])
                            st.session_state.preview_image = Image.open(io.BytesIO(preview_bytes))
                            st.success("✅")
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
    
    # Отображение изображений
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Исходное изображение")
        
        # Показываем превью или оригинал
        if st.session_state.preview_image and st.session_state.channels_count > 3:
            st.image(st.session_state.preview_image, use_column_width=True)
            st.caption(f"RGB композит: {st.session_state.selected_channels}")
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
                st.image(img, use_column_width=True)
                st.caption(f"Размер: {img.width}×{img.height}")
            except:
                st.info("Превью недоступно")
                if st.session_state.channels_count > 3:
                    st.info("💡 Нажмите 'Обновить превью'")
    
    # Классификация
    if st.button("🚀 Классифицировать", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        with st.spinner("Классификация..."):
            try:
                status.text("Отправка данных...")
                progress.progress(25)
                
                files = {"file": (uploaded_file.name, file_bytes)}
                params = {"channels": channels} if channels else {}
                
                resp = requests.post(f"{BACKEND_URL}/classify_all/", files=files, params=params, timeout=300)
                
                progress.progress(75)
                status.text("Обработка...")
                
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.results = data
                    
                    viz_bytes = base64.b64decode(data['visualization'])
                    st.session_state.result_image = Image.open(io.BytesIO(viz_bytes))
                    
                    progress.progress(100)
                    st.success("✅ Готово!")
                else:
                    st.error(f"Ошибка: {resp.status_code}")
            except Exception as e:
                st.error(f"❌ {str(e)}")
            finally:
                progress.empty()
                status.empty()
    
    # Результат
    if st.session_state.result_image:
        with col2:
            st.subheader("Результат")
            st.image(st.session_state.result_image, use_column_width=True)
            st.caption("💡 Легенда слева")
    
    # Скачивание
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📥 Скачать результаты")
        
        cols = st.columns(3)
        
        with cols[0]:
            viz = base64.b64decode(st.session_state.results['visualization'])
            st.download_button("🖼️ PNG", viz, f"{uploaded_file.name.split('.')[0]}_class.png", "image/png")
        
        with cols[1]:
            tif = base64.b64decode(st.session_state.results['geotiff'])
            st.download_button("🗺️ GeoTIFF", tif, f"{uploaded_file.name.split('.')[0]}_class.tif", "image/tiff")
        
        with cols[2]:
            st.download_button("📍 GeoJSON", st.session_state.results['geojson'], 
                             f"{uploaded_file.name.split('.')[0]}_class.geojson", "application/json")
