import streamlit as st
import joblib
import numpy as np

model = joblib.load("divorce_model.pkl")
scaler = joblib.load("scaler.pkl")
questions = joblib.load("question_list.pkl")

st.title("💔 離婚風險預測問卷")
st.write("請根據您對目前婚姻關係的感受，對以下敘述打分（0 = 完全不同意，4 = 完全同意）")

# 範例輸入資料
low_risk = [4 if i not in [5, 6, 20, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] else 0 for i in range(54)]
medium_risk = [2 for _ in range(54)]
high_risk = [0 if i not in [5, 6, 20, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] else 4 for i in range(54)]

def predict_risk(input_list):
    input_array = np.array(input_list).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    return prediction, prob

# 使用者問卷表單
user_input = []
with st.form("form"):
    for q in questions:
        val = st.slider(q, 0, 4, 2)
        user_input.append(val)
    submitted = st.form_submit_button("🔍 根據上面回答預測")

if submitted:
    pred, prob = predict_risk(user_input)
    st.subheader("📊 預測結果（根據填寫問卷）")
    if pred == 1:
        st.error(f"⚠️ 高離婚風險：預測機率為 {prob:.2%}")
    else:
        st.success(f"💡 關係穩定：離婚機率為 {prob:.2%}")

# 範例測試區塊（使用 session_state 記憶狀態）
st.divider()
st.subheader("📌 範例測試")

col1, col2, col3 = st.columns(3)

if col1.button("✅ 測試低風險樣本"):
    st.session_state["test_result"] = ("低風險樣本", *predict_risk(low_risk))

if col2.button("⚖️ 測試中風險樣本"):
    st.session_state["test_result"] = ("中風險樣本", *predict_risk(medium_risk))

if col3.button("❌ 測試高風險樣本"):
    st.session_state["test_result"] = ("高風險樣本", *predict_risk(high_risk))

if "test_result" in st.session_state:
    label, pred, prob = st.session_state["test_result"]
    st.subheader(f"📊 測試結果：{label}")
    if pred == 1:
        st.error(f"⚠️ 高離婚風險：預測機率為 {prob:.2%}")
    else:
        st.success(f"💡 關係穩定：離婚機率為 {prob:.2%}")
