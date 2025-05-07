import streamlit as st
import joblib
import numpy as np

model = joblib.load("divorce_model.pkl")
scaler = joblib.load("scaler.pkl")
questions = joblib.load("question_list.pkl")

st.title("💔 離婚風險預測問卷")
st.write("請根據您對目前婚姻關係的感受，對以下敘述打分（0 = 完全不同意，4 = 完全同意）")

st.markdown("### 🧪 測試範例")
if st.button("低離婚風險（穩定關係）"):
    user_input = [4 if i not in [5, 6, 20, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] else 0 for i in range(len(questions))]
elif st.button("中等離婚風險（普通關係）"):
    user_input = [2 for _ in range(len(questions))]
elif st.button("高離婚風險（高衝突）"):
    user_input = [0 if i not in [5, 6, 20, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] else 4 for i in range(len(questions))]
else:
    user_input = []
    for q in questions:
        val = st.slider(q, 0, 4, 2)
        user_input.append(val)

if st.button("🔍 預測離婚機率") and user_input:
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ 高離婚風險：預測機率為 {prob:.2%}")
    else:
        st.success(f"💡 關係穩定：離婚機率為 {prob:.2%}")