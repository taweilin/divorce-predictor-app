import streamlit as st
import joblib
import numpy as np

model = joblib.load("divorce_model.pkl")
scaler = joblib.load("scaler.pkl")
questions = joblib.load("question_list.pkl")

st.title("💔 離婚風險預測問卷")
st.write("請根據您對目前婚姻關係的感受，對以下敘述打分（0 = 完全不同意，4 = 完全同意）")

user_input = []
for q in questions:
    val = st.slider(q, 0, 4, 2)
    user_input.append(val)

if st.button("🔍 預測離婚機率"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ 高離婚風險：預測機率為 {prob:.2%}")
    else:
        st.success(f"💡 關係穩定：離婚機率為 {prob:.2%}")