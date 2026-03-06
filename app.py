# -*- coding: utf-8 -*-
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import io
import google.generativeai as genai
from PIL import Image

# 1. CẤU HÌNH TRANG PHONG CÁCH GEMINI
st.set_page_config(layout="wide", page_title="Gemini | Đội Định Hóa", page_icon="✨")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .main .block-container { padding-top: 1.5rem; max-width: 900px; margin: auto; }
    
    /* Bong bóng Chat */
    [data-testid="stChatMessage"] { border-radius: 24px; padding: 1.2rem; margin-bottom: 1rem; border: none !important; }
    [data-testid="stChatMessageUser"] { 
        background-color: #f0f4f9 !important; 
        flex-direction: row-reverse !important; 
        margin-left: auto !important; 
        max-width: 80% !important; 
    }
    [data-testid="stChatMessageAssistant"] { 
        background-color: transparent !important; 
    }

    /* Màn hình chào mừng */
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

# 3. LẤY CẤU HÌNH TỪ SECRETS (Đã sửa để đọc trực tiếp, không dùng Fernet)
@st.cache_resource
def get_system_config():
    if "gdrive_service_account" in st.secrets:
        sec = st.secrets["gdrive_service_account"]
        # Xử lý ký tự xuống dòng trong Private Key
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
        
        # Mở file theo tên anh đã đặt
        sh = gc.open("So_tay_GPT_Lich_su")
        
        full_text = "DỮ LIỆU TỪ HỆ THỐNG GOOGLE SHEETS:\n"
        for ws in sh.worksheets():
            data = ws.get_all_records()
            if data:
                df = pd.DataFrame(data)
                full_text += f"\n[Tab: {ws.title}]\n{df.to_string(index=False)}\n"
        return full_text
    except Exception as e:
        return f"⚠️ Không thể đọc Google Sheets: {str(e)}"

# 5. XỬ LÝ AI TRẢ LỜI
def ask_gemini(prompt, files, sheet_data):
    if not config["gemini_api_key"]:
        return "⚠️ Vui lòng bổ sung API Key Gemini vào Secrets."
    
    try:
        genai.configure(api_key=config["gemini_api_key"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Chuẩn bị nội dung gửi đi
        payload = [
            f"Bạn là trợ lý AI chuyên nghiệp của Đội Định Hóa. Hãy trả lời câu hỏi của anh Long.\n\n"
            f"DỮ LIỆU NỘI BỘ (SHEETS):\n{sheet_data}\n\n"
            f"CÂU HỎI HIỆN TẠI: {prompt}"
        ]
        
        # Thêm ảnh nếu có trong bộ nhớ tạm
        for f in files:
            if "image" in f['type']:
                img = Image.open(io.BytesIO(f['content']))
                payload.append(img)
            else:
                payload.append(f"\n[Dữ liệu từ tệp {f['name']}]")
        
        response = model.generate_content(payload)
        return response.text
    except Exception as e:
        return f"❌ Lỗi xử lý AI: {str(e)}"

# 6. GIAO DIỆN SIDEBAR
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=80)
    st.markdown("### ✨ Trợ lý Đội Định Hóa")
    st.divider()
    
    if st.session_state.persistent_files:
        st.markdown("#### 📂 Tệp đã nạp")
        for i, f in enumerate(st.session_state.persistent_files):
            col1, col2 = st.columns([5, 1])
            col1.caption(f"📄 {f['name']}")
            if col2.button("🗑️", key=f"del_{i}"):
                st.session_state.persistent_files.pop(i)
                st.rerun()
    
    if st.button("➕ Cuộc hội thoại mới", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 7. HIỂN THỊ CHAT
if not st.session_state.messages:
    st.markdown('<div class="welcome-text">Chào anh Long,<br>em có thể giúp gì cho anh hôm nay?</div>', unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 8. NHẬP LIỆU & TẢI FILE
st.markdown("---")
uploaded = st.file_uploader("Tải tệp", accept_multiple_files=True, label_visibility="collapsed")
if uploaded:
    for f in uploaded:
        if f.name not in [x['name'] for x in st.session_state.persistent_files]:
            st.session_state.persistent_files.append({"name": f.name, "type": f.type, "content": f.read()})
    st.toast("Đã nạp dữ liệu tệp thành công!")

user_query = st.chat_input("Hỏi em về quy trình hoặc phân tích dữ liệu...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất hệ thống..."):
            ctx_sheet = load_sheet_context()
            res = ask_gemini(user_query, st.session_state.persistent_files, ctx_sheet)
            st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})