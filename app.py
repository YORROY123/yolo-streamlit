import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    import torch
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass
    return YOLO('yolov8m.pt')

st.set_page_config(page_title="YOLOv8 物件辨識", page_icon="🤖")
st.title("🤖 YOLOv8 物件辨識")
st.write("上傳一張圖片，AI 自動辨識裡面的物體！")

uploaded_file = st.file_uploader("選擇圖片", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始圖片", use_container_width=True)

    with st.spinner("辨識中..."):
        model = load_model()
        results = model.predict(np.array(image))
        result_img = results[0].plot()

    st.image(result_img, caption="辨識結果", use_container_width=True)

    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    if labels:
        st.success(f"偵測到：{', '.join(set(labels))}")
    else:
        st.info("沒有偵測到任何物體")