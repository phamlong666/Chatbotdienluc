# -*- coding: utf-8 -*-
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import unicodedata
import datetime
import easyocr
import speech_recognition as sr
import tempfile
import numpy as np
from cryptography.fernet import Fernet
from audio_recorder_streamlit import audio_recorder
import seaborn as sns
import google.generativeai as genai
from fuzzywuzzy import fuzz
import io
from PIL import Image

# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN CHUYÊN NGHIỆP (PHONG CÁCH GEMINI)
st.set_page_config(layout="wide", page_title="Đội Định Hóa AI", page_icon="✨")

# Tùy chỉnh giao diện bằng CSS
st.markdown("""
    <style>
    /* Google-inspired UI */
    .stApp {
        background-color: #ffffff;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    /* Giao diện tin nhắn */
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: none;
        box-shadow: none;
    }
    [data-testid="stChatMessageUser"] {
        background-color: #f0f4f9 !important;
        border-bottom-right-radius: 4px;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: transparent !important;
        border: 1px solid #e3e3e3;
        border-bottom-left-radius: 4px;
    }
    /* Thanh input */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    /* Sidebar */
    .stSidebar {
        background-color: #f8fafd;
    }
    .stSidebar [data-testid="stImage"] {
        border-radius: 50%;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #1f1f1f;
        font-family: 'Google Sans', sans-serif;
    }
    /* Nút bấm */
    .stButton button {
        border-radius: 10px;
        border: 1px solid #dadce0;
        background: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. GIẢI MÃ KHÓA BẢO MẬT
@st.cache_resource
def get_decrypted_all_keys():
    config = {"gemini": None, "gdrive": None}
    if "gdrive_service_account" in st.secrets:
        try:
            sec = st.secrets["gdrive_service_account"]
            master_key = sec.get("encryption_key_for_decryption").encode()
            cipher = Fernet(master_key)
            enc_gemini = sec.get("encrypted_gemini_api_key")
            if enc_gemini:
                config["gemini"] = cipher.decrypt(enc_gemini.encode()).decode()
            enc_g_private = sec.get("encrypted_private_key").encode()
            dec_g_private = cipher.decrypt(enc_g_private).decode()
            config["gdrive"] = {
                "type": sec.get("type", "service_account"),
                "project_id": sec.get("project_id"),
                "private_key_id": sec.get("private_key_id"),
                "private_key": dec_g_private,
                "client_email": sec.get("client_email"),
                "client_id": sec.get("client_id"),
                "auth_uri": sec.get("auth_uri"),
                "token_uri": sec.get("token_uri"),
                "auth_provider_x509_cert_url": sec.get("auth_provider_x509_cert_url"),
                "client_x509_cert_url": sec.get("client_x509_cert_url")
            }
        except Exception as e:
            st.error(f"Lỗi bảo mật: {e}")
    return config

secrets_data = get_decrypted_all_keys()

# 3. KẾT NỐI GOOGLE SHEETS
def get_sheets_connection():
    if secrets_data["gdrive"]:
        try:
            creds = Credentials.from_service_account_info(secrets_data["gdrive"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
            return gspread.authorize(creds)
        except: return None
    return None

gc = get_sheets_connection()
# Đường link Sheet anh gửi
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/13MqQzvV3Mf9bLOAXwICXclYVQ-8WnvBDPAR8VJfOGJg/edit"

# 4. CÔNG CỤ XỬ LÝ
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

@st.cache_data(ttl=300)
def load_data():
    if not gc: return {}
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        data = {}
        for ws in sh.worksheets():
            df = pd.DataFrame(ws.get_all_records())
            df.columns = [str(c).strip() for c in df.columns]
            data[ws.title] = df
        return data
    except: return {}

def query_gemini(prompt, files=None):
    if not secrets_data["gemini"]: return "Cần cấu hình API Gemini."
    try:
        genai.configure(api_key=secrets_data["gemini"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        content = [prompt]
        if files: content.extend(files)
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Lỗi AI: {str(e)}"

# 5. GIAO DIỆN CHÍNH
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar nâng cấp
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=120)
    st.markdown("### ✨ Đội Định Hóa AI")
    st.caption("Trợ lý phân tích dữ liệu chuyên sâu")
    
    st.divider()
    # Khu vực Upload File
    st.markdown("### 📄 Tài liệu phân tích")
    uploaded_files = st.file_uploader("Tải Word, Excel, PDF hoặc Ảnh", 
                                    type=["pdf", "xlsx", "docx", "png", "jpg"], 
                                    accept_multiple_files=True)
    
    if st.button("🗑 Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Hiển thị tiêu đề chào mừng nếu chưa có chat
if not st.session_state.messages:
    st.markdown("<h2 style='text-align: center; margin-top: 100px; color: #444;'>Chào anh Long, hôm nay em có thể giúp gì cho anh?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Anh có thể hỏi về nghiệp vụ hoặc tải file lên để em báo cáo nhé.</p>", unsafe_allow_html=True)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg: st.dataframe(msg["df"])

# Xử lý input
u_input = st.chat_input("Hỏi em bất cứ điều gì...")

if u_input:
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"): st.markdown(u_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            all_data = load_data()
            handled = False
            
            # --- LUỒNG 1: TÌM TRONG SAMPLE_QUESTIONS (Hỏi-Trả lời) ---
            # Anh lưu ý: GID 2107842233 tương ứng với sheet "Hỏi-Trả lời" hoặc "sample_questions"
            df_qa = all_data.get("Hỏi-Trả lời", all_data.get("sample_questions", pd.DataFrame()))
            
            if not df_qa.empty:
                best_ans = None
                max_score = 0
                for _, row in df_qa.iterrows():
                    score = fuzz.token_set_ratio(normalize_text(u_input), normalize_text(str(row.get("Câu hỏi", ""))))
                    if score > 75: # Ngưỡng khớp
                        best_ans = row.get("Câu trả lời", "")
                        break
                
                if best_ans:
                    st.markdown(best_ans)
                    st.session_state.messages.append({"role": "assistant", "content": best_ans})
                    handled = True

            # --- LUỒNG 2: PHÂN TÍCH FILE (NẾU CÓ FILE UPLOAD) ---
            if not handled and uploaded_files:
                file_context = "Người dùng đã tải lên các tệp sau để phân tích. Hãy dựa vào đó trả lời: "
                # Xử lý file tại đây (đơn giản hóa qua prompt)
                ans = query_gemini(f"{file_context} \n Câu hỏi: {u_input}")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                handled = True

            # --- LUỒNG 3: TỰ ĐỘNG CHUYỂN GEMINI (NẾU KHÔNG CÓ TRONG SHEET) ---
            if not handled:
                # Nếu hỏi về số liệu có sẵn (KPI, CBCNV) thì hiện bảng trước
                norm_u = normalize_text(u_input)
                if any(k in norm_u for k in ["kpi", "nhân sự", "nhân viên"]):
                    df_kpi = all_data.get("KPI", pd.DataFrame())
                    if not df_kpi.empty:
                        res = "Đây là dữ liệu em tìm thấy trong hệ thống:"
                        st.markdown(res)
                        st.dataframe(df_kpi)
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": df_kpi})
                        handled = True
                
                # Cuối cùng là hỏi Gemini kiến thức tổng quát
                if not handled:
                    ans = query_gemini(u_input)
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})