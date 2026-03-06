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

# 1. CẤU HÌNH TRANG PHONG CÁCH GEMINI (TỐI GIẢN & HIỆN ĐẠI)
st.set_page_config(layout="wide", page_title="Gemini | Đội Định Hóa", page_icon="✨")

# CSS để tạo giao diện Gemini chuẩn
st.markdown("""
    <style>
    /* Tổng thể App */
    .stApp {
        background-color: #ffffff;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
        margin: auto;
    }
    
    /* Giao diện Chat */
    [data-testid="stChatMessage"] {
        border-radius: 24px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        border: none !important;
    }

    /* Tin nhắn Người dùng (Phải) */
    [data-testid="stChatMessageUser"] {
        background-color: #f0f4f9 !important;
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        max-width: 80% !important;
    }

    /* Tin nhắn Assistant (Trái) */
    [data-testid="stChatMessageAssistant"] {
        background-color: transparent !important;
        margin-right: auto !important;
        max-width: 100% !important;
        padding-left: 0 !important;
    }

    /* Khung nhập liệu (Chat Input) */
    .stChatInputContainer {
        border-radius: 32px !important;
        border: 1px solid #dee2e6 !important;
        background-color: #f8fafd !important;
        padding: 5px 15px !important;
    }

    /* Tiêu đề & Chào mừng */
    .welcome-text {
        font-family: 'Google Sans', Arial, sans-serif;
        font-size: 40px;
        font-weight: 500;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 15vh;
    }
    
    .sidebar-text {
        font-size: 14px;
        color: #444;
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
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/13MqQzvV3Mf9bLOAXwICXclYVQ-8WnvBDPAR8VJfOGJg/edit"

# 4. HÀM XỬ LÝ DỮ LIỆU & AI
@st.cache_data(ttl=60)
def load_full_context():
    if not gc: return ""
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        context = "DỮ LIỆU NỘI BỘ ĐỘI ĐỊNH HÓA:\n"
        for ws in sh.worksheets():
            df = pd.DataFrame(ws.get_all_records())
            context += f"\n--- Bảng: {ws.title} ---\n{df.to_string(index=False)}\n"
        return context
    except: return ""

def query_gemini_pro(prompt, files=None, sheet_context=""):
    if not secrets_data["gemini"]: return "⚠️ Chưa có API Key."
    try:
        genai.configure(api_key=secrets_data["gemini"])
        # Sử dụng model Pro để xử lý context lớn
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        full_prompt = f"""
        Bạn là trợ lý AI thông minh của Đội Định Hóa. 
        Sử dụng dữ liệu sau đây để trả lời câu hỏi của người dùng một cách chính xác nhất.
        Nếu cần vẽ biểu đồ, hãy mô tả cách vẽ hoặc cung cấp dữ liệu định dạng bảng.
        
        {sheet_context}
        
        Câu hỏi của người dùng: {prompt}
        """
        
        content = [full_prompt]
        if files: content.extend(files)
        
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# 5. GIAO DIỆN CHÍNH
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar nâng cấp
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=80)
    st.markdown("### ✨ Gemini Đội Định Hóa")
    st.divider()
    
    st.markdown("#### 📂 Tải tệp phân tích")
    uploaded_files = st.file_uploader("Thả Word, Excel, PDF hoặc Ảnh tại đây", 
                                    type=["pdf", "xlsx", "docx", "png", "jpg"], 
                                    accept_multiple_files=True)
    
    st.divider()
    if st.button("➕ Cuộc hội thoại mới", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# HIỂN THỊ MÀN HÌNH CHÀO MỪNG (GIỐNG GEMINI)
if not st.session_state.messages:
    st.markdown('<div class="welcome-text">Chào anh Long,<br>hôm nay em có thể giúp gì cho anh?</div>', unsafe_allow_html=True)
    
    # Gợi ý nhanh
    cols = st.columns(3)
    with cols[0]:
        if st.button("📊 Báo cáo KPI tháng này"): u_input = "Hãy phân tích KPI tháng này của đội"
    with cols[1]:
        if st.button("👥 Danh sách nhân sự"): u_input = "Cho anh xem danh sách nhân viên"
    with cols[2]:
        if st.button("📝 Tóm tắt tài liệu"): u_input = "Hãy tóm tắt file anh vừa gửi"

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Khu vực nhập liệu
u_input = st.chat_input("Nhập câu hỏi hoặc dán link tài liệu...")

if u_input:
    # Lưu và hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"):
        st.markdown(u_input)

    # Xử lý trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            # Lấy toàn bộ dữ liệu từ Sheet làm ngữ cảnh cho AI
            context = load_full_context()
            
            # Xử lý File nếu có
            gemini_files = []
            if uploaded_files:
                for f in uploaded_files:
                    if f.type in ["image/png", "image/jpeg"]:
                        gemini_files.append(Image.open(f))
                    # Các loại file khác sẽ được Gemini xử lý qua text nếu cần (tối giản ở đây)

            # Gọi Gemini xử lý tổng hợp
            answer = query_gemini_pro(u_input, files=gemini_files, sheet_context=context)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})