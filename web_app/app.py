import streamlit as st
import torch
from PIL import Image
from predict import predict_image # ดึงฟังก์ชันมาจากไฟล์ predict ของคุณ
from predict import predict_image, class_names

st.title("Astro Quality Monitoring System 🔭")

uploaded_file = st.file_uploader("เลือกภาพดาราศาสตร์ที่นี่...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='ภาพที่คุณอัปโหลด', use_column_width=True)
    
    # ทำนายผล
    st.write("กำลังประเมินคุณภาพ...")
    probs, indices = predict_image(uploaded_file) # ใช้ฟังก์ชันเดิมที่คุณเขียนไว้
    
    # แสดงผลอันดับ 1
    st.success(f"ผลการทำนาย: {class_names[indices[0]]} ({probs[0]*100:.2f}%)")
    
    # แสดงตาราง 5 อันดับ
    st.write("---")
    st.write("### รายละเอียดความมั่นใจ 5 อันดับ:")
    for i in range(5):
        st.write(f"**{class_names[indices[i]]}**: {probs[i]*100:.2f}%")