import streamlit as st
from PIL import Image
import tempfile
from ultralytics import YOLO
import os

# โหลดโมเดล YOLO
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# UI ส่วนบน
st.title("🧍‍♂️ People Counter with YOLOv11n + Streamlit")
st.markdown("อัปโหลดภาพเพื่อให้ AI ตรวจจับและนับจำนวนคนในภาพ")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # เปิดภาพด้วย PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 ภาพที่อัปโหลด", use_container_width=True)

    # บันทึกไฟล์ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_image_path = temp_file.name

    # รัน object detection
    results = model(source=temp_image_path, save=True, conf=0.3, verbose=False)

    # ดึงผลลัพธ์จากโมเดล
    for result in results:
        class_ids = result.boxes.cls.int()
        class_names = [result.names[class_id.item()] for class_id in class_ids]
        person_count = class_names.count("person")

        st.success(f"🧍‍♀️ ตรวจพบจำนวนคนทั้งหมด: **{person_count}** คน")

        # แสดงภาพที่มีการวาด Bounding Box โดยตรง (ไม่ต้องโหลดจาก disk)
        result_img_array = result.plot()  # Numpy array with bounding boxes
        result_img = Image.fromarray(result_img_array)
        st.image(result_img, caption="📍 ผลลัพธ์หลังตรวจจับ",  use_container_width=True)

    # ลบไฟล์ชั่วคราว
    os.remove(temp_image_path)
