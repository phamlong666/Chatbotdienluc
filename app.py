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

# CSS để tạo giao diện Gemini chuẩn và quản lý tệp
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .main .block-container { padding-top: 1rem; max-width: 950px; margin: auto; }
    
    /* Giao diện Chat */
    [data-testid="stChatMessage"] { border-radius: 24px; padding: 1.2rem; margin-bottom: 1rem; border: none !important; }
    [data-testid="stChatMessageUser"] { background-color: #f0f4f9 !important; flex-direction: row-reverse !important; margin-left: auto !important; max-width: 80% !important; }
    [data-testid="stChatMessageAssistant"] { background-color: transparent !important; margin-right: auto !important; max-width: 100% !important; padding-left: 0 !important; }

    /* Khung nhập liệu */
    .stChatInputContainer { border-radius: 32px !important; border: 1px solid #dee2e6 !important; background-color: #f8fafd !important; }

    /* Màn hình chào mừng */
    .welcome-text {
        font-family: 'Google Sans', Arial, sans-serif;
        font-size: 38px;
        font-weight: 500;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 10vh;
        margin-bottom: 20px;
    }
    
    /* Danh sách tệp đã tải */
    .file-chip {
        display: inline-flex;
        align-items: center;
        background-color: #f1f3f4;
        border-radius: 16px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 13px;
        border: 1px solid #dadce0;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. KHỞI TẠO STATE (Lưu trữ bền vững trong session)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "persistent_files" not in st.session_state:
    st.session_state.persistent_files = [] # Lưu danh sách dict: {name, type, content}

# 3. GIẢI MÃ KHÓA BẢO MẬT
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
                "type": "service_account",
                "project_id": sec.get("project_id"),
                "private_key": dec_g_private,
                "client_email": sec.get("client_email"),
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        except: pass
    return config

secrets_data = get_decrypted_all_keys()

# 4. KẾT NỐI DỮ LIỆU
def get_sheets_connection():
    if secrets_data["gdrive"]:
        try:
            creds = Credentials.from_service_account_info(secrets_data["gdrive"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
            return gspread.authorize(creds)
        except: return None
    return None

gc = get_sheets_connection()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/13MqQzvV3Mf9bLOAXwICXclYVQ-8WnvBDPAR8VJfOGJg/edit"

@st.cache_data(ttl=60)
def load_full_context():
    if not gc: return ""
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        context = "DỮ LIỆU NỘI BỘ GOOGLE SHEETS:\n"
        for ws in sh.worksheets():
            df = pd.DataFrame(ws.get_all_records())
            context += f"\n[Bảng {ws.title}]\n{df.to_string(index=False)}\n"
        return context
    except: return ""

def query_gemini_pro(prompt, files_data, sheet_context):
    if not secrets_data["gemini"]: return "⚠️ Chưa có API Key."
    try:
        genai.configure(api_key=secrets_data["gemini"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Xây dựng context từ tệp đã tải
        file_summary = ""
        gemini_payload = []
        
        if files_data:
            file_summary = "DỮ LIỆU TỪ TỆP ĐÃ TẢI LÊN:\n"
            for f in files_data:
                file_summary += f"- Tên tệp: {f['name']}\n"
                # Nếu là ảnh, đưa vào payload trực tiếp
                if "image" in f['type']:
                    gemini_payload.append(Image.open(io.BytesIO(f['content'])))
        
        full_instructions = f"""
        Bạn là trợ lý AI cao cấp của Đội Định Hóa.
        {sheet_context}
        {file_summary}
        Nhiệm vụ: Trả lời câu hỏi dựa trên dữ liệu Google Sheets và các tệp đính kèm. 
        Nếu cần vẽ biểu đồ, hãy hướng dẫn chi tiết hoặc cung cấp bảng dữ liệu.
        Câu hỏi: {prompt}
        """
        
        content = [full_instructions] + gemini_payload
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# 5. GIAO DIỆN CHÍNH
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=70)
    st.markdown("### ✨ Gemini Đội Định Hóa")
    st.divider()
    
    # Nút xóa tệp có xác nhận
    if st.session_state.persistent_files:
        st.markdown("#### 📂 Tệp đang sử dụng")
        for i, f in enumerate(st.session_state.persistent_files):
            col1, col2 = st.columns([4, 1])
            col1.caption(f"📄 {f['name']}")
            if col2.button("🗑️", key=f"del_{i}"):
                st.session_state.file_to_delete = i
                st.rerun()
        
        if "file_to_delete" in st.session_state:
            st.warning(f"Xóa tệp {st.session_state.persistent_files[st.session_state.file_to_delete]['name']}?")
            c1, c2 = st.columns(2)
            if c1.button("Đồng ý", use_container_width=True):
                st.session_state.persistent_files.pop(st.session_state.file_to_delete)
                del st.session_state.file_to_delete
                st.rerun()
            if c2.button("Hủy", use_container_width=True):
                del st.session_state.file_to_delete
                st.rerun()

    st.divider()
    if st.button("➕ Hội thoại mới", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Màn hình chào mừng
if not st.session_state.messages:
    st.markdown('<div class="welcome-text">Chào anh Long,<br>hôm nay em có thể giúp gì cho anh?</div>', unsafe_allow_html=True)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# KHU VỰC NHẬP LIỆU & TẢI FILE (Nằm cuối trang)
st.markdown("---")
# Upload file ngay cửa sổ chat (nhỏ gọn)
new_files = st.file_uploader("📎 Đính kèm tệp để phân tích (PDF, Excel, Ảnh...)", 
                             accept_multiple_files=True, label_visibility="collapsed")

if new_files:
    for f in new_files:
        # Kiểm tra xem tệp đã tồn tại chưa để tránh trùng lặp
        if f.name not in [x['name'] for x in st.session_state.persistent_files]:
            st.session_state.persistent_files.append({
                "name": f.name,
                "type": f.type,
                "content": f.read()
            })
    st.success(f"Đã nạp {len(new_files)} tệp vào bộ nhớ.")

u_input = st.chat_input("Hỏi em hoặc yêu cầu lập báo cáo...")

if u_input:
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"):
        st.markdown(u_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý dữ liệu tổng hợp..."):
            context = load_full_context()
            answer = query_gemini_pro(u_input, st.session_state.persistent_files, context)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})