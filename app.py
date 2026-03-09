# -*- coding: utf-8 -*-
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import io
from PIL import Image
import google.generativeai as genai

# =====================================================
# 1. CẤU HÌNH TRANG
# =====================================================

st.set_page_config(
    page_title="Trợ lý AI Đội Định Hóa",
    page_icon="✨",
    layout="wide"
)

# =====================================================
# 2. CSS GIAO DIỆN CHAT (TRÁI / PHẢI)
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
}

.main .block-container{
    max-width: 900px;
    margin:auto;
}

/* Bong bóng chat chung */
[data-testid="stChatMessage"]{
    border-radius:18px;
    padding:14px;
    margin-bottom:10px;
    width:fit-content;
    max-width:80%;
}

/* USER */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    margin-left:auto;
    background:#f1f3f4;
}

/* BOT */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    margin-right:auto;
    background:#ffffff;
    border:1px solid #e3e3e3;
}

.welcome{
    text-align:center;
    font-size:36px;
    font-weight:500;
    margin-top:8vh;
    background: linear-gradient(90deg,#4285f4,#9b72cb,#d96570);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = []

# =====================================================
# 4. ĐỌC CẤU HÌNH TỪ SECRETS
# =====================================================

@st.cache_resource
def load_config():

    if "gdrive_service_account" not in st.secrets:
        return None

    sec = st.secrets["gdrive_service_account"]

    raw_key = sec.get("encrypted_private_key")

    private_key = raw_key.replace("\\n", "\n") if raw_key else None

    config = {
        "gemini_api_key": sec.get("encrypted_gemini_api_key"),
        "gdrive":{
            "type":"service_account",
            "project_id":sec.get("project_id"),
            "private_key_id":sec.get("private_key_id"),
            "private_key":private_key,
            "client_email":sec.get("client_email"),
            "token_uri":sec.get("token_uri")
        }
    }

    return config

config = load_config()

# =====================================================
# 5. KẾT NỐI GOOGLE SHEETS
# =====================================================

@st.cache_data(ttl=600)
def load_sheet():

    if not config:
        return ""

    try:

        creds = Credentials.from_service_account_info(
            config["gdrive"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )

        client = gspread.authorize(creds)

        sh = client.open("So_tay_GPT_Lich_su")

        text = "DỮ LIỆU GOOGLE SHEET:\n"

        for ws in sh.worksheets():

            data = ws.get_all_records()

            if data:

                df = pd.DataFrame(data)

                text += f"\n[{ws.title}]\n"
                text += df.to_string(index=False)
                text += "\n"

        return text

    except Exception as e:

        return f"Lỗi đọc Google Sheet: {e}"

# =====================================================
# 6. KHỞI TẠO GEMINI
# =====================================================

@st.cache_resource
def load_model():

    if not config:
        return None

    api_key = config.get("gemini_api_key")

    if not api_key:
        return None

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash"
    )

    return model

model = load_model()

# =====================================================
# 7. HÀM CHAT GEMINI
# =====================================================

def ask_gemini(question, files, sheet_context):

    if not model:
        return "⚠️ Chưa cấu hình Gemini API"

    prompt = f"""
Bạn là trợ lý AI của Đội Quản Lý Điện Lực Khu Vực Định Hóa.

Dữ liệu nội bộ:
{sheet_context}

Câu hỏi:
{question}
"""

    inputs = [prompt]

    for f in files:

        if "image" in f["type"]:

            img = Image.open(io.BytesIO(f["content"]))
            inputs.append(img)

    try:

        res = model.generate_content(inputs)

        return res.text

    except Exception as e:

        return f"❌ Lỗi AI: {e}"

# =====================================================
# 8. SIDEBAR
# =====================================================

with st.sidebar:

    st.image(
        "https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png",
        width=80
    )

    st.markdown("### Trợ lý AI Đội Định Hóa")

    if st.button("➕ Hội thoại mới"):

        st.session_state.messages = []
        st.session_state.files = []

        st.rerun()

# =====================================================
# 9. WELCOME
# =====================================================

if not st.session_state.messages:

    st.markdown(
        '<div class="welcome">Chào anh Long, em sẵn sàng hỗ trợ!</div>',
        unsafe_allow_html=True
    )

# =====================================================
# 10. HIỂN THỊ LỊCH SỬ CHAT
# =====================================================

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# =====================================================
# 11. UPLOAD FILE
# =====================================================

uploaded_files = st.file_uploader(
    "Tải file",
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:

    for f in uploaded_files:

        if f.name not in [x["name"] for x in st.session_state.files]:

            st.session_state.files.append({
                "name":f.name,
                "type":f.type,
                "content":f.read()
            })

    st.toast("Đã nạp file")

# =====================================================
# 12. INPUT CHAT
# =====================================================

question = st.chat_input("Hỏi AI về dữ liệu, KPI, thời tiết...")

if question:

    st.session_state.messages.append({
        "role":"user",
        "content":question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("AI đang suy nghĩ..."):

            sheet_data = load_sheet()

            answer = ask_gemini(
                question,
                st.session_state.files,
                sheet_data
            )

            st.markdown(answer)

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })
