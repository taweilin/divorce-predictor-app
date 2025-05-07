import streamlit as st
import numpy as np
import joblib

# 載入模型、scaler、題目
model = joblib.load("divorce_model.pkl")
scaler = joblib.load("scaler.pkl")
questions = joblib.load("question_list.pkl")

st.title("離婚風險預測系統")

st.markdown("請根據實際情況回答下列問題（0：完全不同意，4：完全同意）")

# 範例資料
low_risk_sample = [3]*54
mid_risk_sample = [2]*54
high_risk_sample = [0 if i not in [5, 6, 20, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] else 4 for i in range(54)]

# 範例按鈕
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("低風險樣本"):
        for i, val in enumerate(low_risk_sample):
            st.session_state[f"Q{i+1}"] = val
with col2:
    if st.button("中風險樣本"):
        for i, val in enumerate(mid_risk_sample):
            st.session_state[f"Q{i+1}"] = val
with col3:
    if st.button("高風險樣本"):
        for i, val in enumerate(high_risk_sample):
            st.session_state[f"Q{i+1}"] = val

# 顯示問題與輸入
user_input = []
for i, q in enumerate(questions):
    val = st.slider(f"{i+1}. {q}", 0, 4, key=f"Q{i+1}")
    user_input.append(val)

# 預測函式
def predict_risk(input_list):
    input_array = np.array(input_list).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  # 1 = 離婚（經修正標籤）
    return prediction, prob

# 按鈕觸發預測
if st.button("預測離婚風險"):
    pred, prob = predict_risk(user_input)
    prob_percent = round(prob * 100, 2)
    if pred == 1:
        st.success(f"關係穩定：離婚機率為 {prob_percent:.2f}%")
    else:
        st.error(f"高離婚風險：預測機率為 {prob_percent:.2f}%")
