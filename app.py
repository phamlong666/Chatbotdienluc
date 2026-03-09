# -*- coding: utf-8 -*-
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import io
import google.generativeai as genai
from PIL import Image

# 1. CẤU HÌNH TRANG & GIAO DIỆN (Đã sửa chia 2 bên)
st.set_page_config(layout="wide", page_title="Gemini | Đội Định Hóa", page_icon="✨")

st.markdown("""
    <style>
    /* Tổng thể khung chat */
    .stApp { background-color: #ffffff; }
    .main .block-container { padding-top: 1.5rem; max-width: 900px; margin: auto; }

    /* Định dạng chung cho bong bóng chat */
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: none !important;
        display: flex !important;
        width: fit-content !important;
        max-width: 85% !important;
    }

    /* USER (Bên phải) */
    [data-testid="stChatMessageUser"] {
        background-color: #f0f4f9 !important;
        margin-left: auto !important;
        flex-direction: row-reverse !important;
        text-align: left;
    }

    /* ASSISTANT (Bên trái) */
    [data-testid="stChatMessageAssistant"] {
        background-color: #ffffff !important;
        margin-right: auto !important;
        flex-direction: row !important;
        border: 1px solid #e3e3e3 !important;
        text-align: left;
    }

    .welcome-text {
        font-family: 'Google Sans', sans-serif;
        font-size: 38px;
        font-weight: 500;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 8vh;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. KHỞI TẠO STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "persistent_files" not in st.session_state:
    st.session_state.persistent_files = []

# 3. LẤY CẤU HÌNH TỪ SECRETS
@st.cache_resource
def get_system_config():
    if "gdrive_service_account" in st.secrets:
        sec = st.secrets["gdrive_service_account"]
        raw_key = sec.get("encrypted_private_key")
        formatted_key = raw_key.replace("\\n", "\n") if raw_key else None
        return {
            "gemini_api_key": sec.get("encrypted_gemini_api_key"),
            "gdrive": {
                "type": "service_account",
                "project_id": sec.get("project_id"),
                "private_key_id": sec.get("private_key_id"),
                "private_key": formatted_key,
                "client_email": sec.get("client_email"),
                "token_uri": sec.get("token_uri"),
            }
        }
    return None

config = get_system_config()

# 4. KẾT NỐI GOOGLE SHEETS
def load_sheet_context():
    if not config or not config["gdrive"]["private_key"]:
        return "⚠️ Thiếu cấu hình Service Account trong Secrets."
    try:
        creds = Credentials.from_service_account_info(config["gdrive"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        # Tên file Google Sheet của anh
        sh = gc.open("So_tay_GPT_Lich_su")
        full_text = "DỮ LIỆU TỪ HỆ THỐNG GOOGLE SHEETS:\n"
        for ws in sh.worksheets():
            data = ws.get_all_records()
            if data:
                df = pd.DataFrame(data)
                full_text += f"\n[Tab: {ws.title}]\n{df.to_string(index=False)}\n"
        return full_text
    except Exception as e:
        return f"⚠️ Lỗi đọc Sheet: {str(e)}"

# 5. XỬ LÝ AI TRẢ LỜI
def ask_gemini(prompt, files, sheet_data):
    api_key = config["gemini_api_key"]
    if not api_key: return "⚠️ Thiếu API Key trong cấu hình."
    
    genai.configure(api_key=api_key)
    
    # Chuẩn bị nội dung gửi đi
    system_instruction = (
        f"Bạn là trợ lý AI chuyên nghiệp của Đội Định Hóa. Hãy trả lời câu hỏi của anh Long.\n\n"
        f"DỮ LIỆU TRONG GOOGLE SHEET CỦA ANH LONG:\n{sheet_data}\n\n"
        f"LƯU Ý: Nếu câu hỏi về tin tức mới nhất hoặc thời tiết, hãy ưu tiên dùng Google Search."
    )
    
    full_prompt = [system_instruction, f"CÂU HỎI HIỆN TẠI: {prompt}"]
    
    # Đính kèm tệp nếu có
    for f in files:
        if "image" in f['type']:
            img = Image.open(io.BytesIO(f['content']))
            full_prompt.append(img)
        else:
            full_prompt.append(f"\n[Dữ liệu tệp đính kèm {f['name']}]")

    try:
        # Cố gắng sử dụng model với Google Search (cú pháp mới nhất)
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            tools=[{"google_search_retrieval": {}}] 
        )
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        # FALLBACK: Nếu lỗi Search, dùng model cơ bản để đảm bảo luôn có phản hồi
        try:
            model_basic = genai.GenerativeModel(model_name='gemini-1.5-flash')
            response = model_basic.generate_content(full_prompt)
            return response.text
        except Exception as e2:
            return f"❌ Lỗi hệ thống AI: {str(e2)}"

# 6. GIAO DIỆN CHÍNH
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=80)
    st.markdown("### ✨ Trợ lý Đội Định Hóa")
    if st.button("➕ Hội thoại mới"):
        st.session_state.messages = []
        st.session_state.persistent_files = [] # Xóa tệp cũ khi tạo hội thoại mới
        st.rerun()

# Hiển thị lời chào
if not st.session_state.messages:
    st.markdown('<div class="welcome-text">Chào anh Long,<br>em đã sẵn sàng hỗ trợ anh!</div>', unsafe_allow_html=True)

# Hiển thị lịch sử chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]): 
        st.markdown(m["content"])

st.markdown("---")
# Khu vực tải tệp
uploaded = st.file_uploader("Tải tệp", accept_multiple_files=True, label_visibility="collapsed")
if uploaded:
    for f in uploaded:
        if f.name not in [x['name'] for x in st.session_state.persistent_files]:
            file_bytes = f.read()
            st.session_state.persistent_files.append({"name": f.name, "type": f.type, "content": file_bytes})
    st.toast("Đã nạp tệp!")

# Ô nhập câu hỏi
user_query = st.chat_input("Hỏi em về KPI hoặc thời tiết...")

if user_query:
    # Lưu và hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"): 
        st.markdown(user_query)

    # Chatbot phản hồi
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất dữ liệu..."):
            ctx_sheet = load_sheet_context()
            res = ask_gemini(user_query, st.session_state.persistent_files, ctx_sheet)
            st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})