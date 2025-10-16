
import streamlit as st
import numpy as np
import joblib

# Load mô hình và scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("stacking.pkl")  # hoặc linear.pkl, mlp.pkl...

# Tiêu đề ứng dụng
st.set_page_config(page_title="Đánh giá chất lượng rượu vang", layout="centered")
st.title("🍷 Đánh giá chất lượng rượu vang")

st.markdown("### Nhập các thông số hóa học (giá trị từ 0 đến 15):")

# Tạo các input cho người dùng
features = []
feature_names = [
    "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
    "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density",
    "pH", "Sulphates", "Alcohol"
]

for name in feature_names:
    value = st.slider(name, 0.0, 15.0, 7.5)
    features.append(value)

# Dự đoán khi người dùng nhấn nút
if st.button("Dự đoán chất lượng"):
    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    # Hiển thị nhãn phân loại
    st.success(f"✅ Chất lượng rượu: **{prediction}**")
