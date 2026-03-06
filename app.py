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

# --- THƯ VIỆN CHO SEMANTIC SEARCH ---
from sentence_transformers import SentenceTransformer, util
import torch

# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN CHUNG
st.set_page_config(layout="wide", page_title="AI Đội Định Hóa", page_icon="🤖")

# CSS Tùy chỉnh để giao diện chia 2 bên rõ ràng
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 15px;
        padding: 10px;
    }
    /* Người dùng (User): Nằm bên phải */
    [data-testid="stChatMessageUser"] {
        background-color: #e3f2fd !important;
        flex-direction: row-reverse;
        text-align: right;
    }
    /* Trợ lý (Bot): Nằm bên trái */
    [data-testid="stChatMessageAssistant"] {
        background-color: #f5f5f5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cấu hình tiếng Việt cho biểu đồ
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# 2. GIẢI MÃ KHÓA BẢO MẬT (LOGIC CŨ CỦA ANH)
@st.cache_resource
def get_decrypted_all_keys():
    config = {"gemini": None, "gdrive": None}
    if "gdrive_service_account" in st.secrets:
        try:
            sec = st.secrets["gdrive_service_account"]
            master_key = sec.get("encryption_key_for_decryption").encode()
            cipher = Fernet(master_key)
            
            # Giải mã API Key Gemini (AI Studio)
            enc_gemini = sec.get("encrypted_gemini_api_key")
            if enc_gemini:
                config["gemini"] = cipher.decrypt(enc_gemini.encode()).decode()
            
            # Giải mã Private Key của Google Drive
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
            st.error(f"Lỗi giải mã bảo mật: {e}")
    return config

secrets_data = get_decrypted_all_keys()

# 3. KHỞI TẠO CÔNG CỤ AI
@st.cache_resource
def init_ai_tools():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "semantic_model": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device),
        "ocr_reader": easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
    }

with st.spinner("🤖 Đang kết nối trí tuệ nhân tạo..."):
    ai_tools = init_ai_tools()

# 4. KẾT NỐI GOOGLE SHEETS
def get_sheets_connection():
    if secrets_data["gdrive"]:
        try:
            creds = Credentials.from_service_account_info(secrets_data["gdrive"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
            return gspread.authorize(creds)
        except: return None
    return None

gc = get_sheets_connection()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/13MqQzvV3Mf9bLOAXwICXclYVQ-8WnvBDPAR8VJfOGJg/edit"

# 5. CÁC HÀM TRỢ GIÚP
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

@st.cache_data(ttl=300)
def load_all_sheets():
    if not gc: return {}
    try:
        sh = gc.open_by_url(SPREADSHEET_URL)
        return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sh.worksheets()}
    except: return {}

def semantic_search(query, df, q_col, a_col, threshold=0.5):
    if df.empty or q_col not in df.columns: return None, 0
    questions = df[q_col].astype(str).tolist()
    doc_embs = ai_tools["semantic_model"].encode(questions, convert_to_tensor=True)
    query_emb = ai_tools["semantic_model"].encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, doc_embs)[0]
    best_idx = torch.argmax(cos_scores).item()
    score = float(cos_scores[best_idx])
    if score > threshold:
        return df.iloc[best_idx][a_col], score * 100
    return None, 0

# 6. GIAO DIỆN CHAT VÀ QUẢN LÝ HỘI THOẠI
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_text" not in st.session_state:
    st.session_state.voice_text = None

# SIDEBAR
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=120)
    st.title("Trợ lý Đội Định Hóa")
    st.divider()
    
    # Nút Xóa/Lưu
    if st.button("🗑 Xóa lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("💾 Lưu hội thoại", use_container_width=True):
        st.success("Đã lưu hội thoại!")

    st.divider()
    # Giọng nói
    audio_val = audio_recorder(text="Bấm để nói", icon_size="2x")
    if audio_val:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_val)
            tmp_path = tmp.name
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            try:
                audio = r.record(source)
                st.session_state.voice_text = r.recognize_google(audio, language="vi-VN")
            except: st.error("Không nghe rõ...")
        os.remove(tmp_path)

# HIỂN THỊ CHAT (Giao diện chia 2 bên)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fig" in msg: st.pyplot(msg["fig"])
        if "df" in msg: st.dataframe(msg["df"])

# NHẬN ĐẦU VÀO
u_input = st.session_state.voice_text if st.session_state.voice_text else st.chat_input("Hỏi về KPI, nhân sự, TBA hoặc kiến thức chung...")
st.session_state.voice_text = None

if u_input:
    # 1. Hiển thị User nhắn (Bên phải)
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"):
        st.markdown(u_input)

    # 2. Xử lý Trợ lý trả lời (Bên trái)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất..."):
            all_data = load_all_sheets()
            handled = False
            norm_u = normalize_text(u_input)

            # --- LUỒNG 1: TRUY VẤN GOOGLE SHEETS ---
            # (KPI, CBCNV, Lãnh đạo xã, TBA...)
            if any(k in norm_u for k in ["kpi", "chỉ số"]):
                df_kpi = all_data.get("KPI", pd.DataFrame())
                if not df_kpi.empty:
                    st.dataframe(df_kpi)
                    res = "Dữ liệu KPI anh cần ạ."
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": df_kpi})
                    handled = True

            elif any(k in norm_u for k in ["cbcnv", "nhân viên", "nhân sự"]):
                df_cb = all_data.get("CBCNV", pd.DataFrame())
                if not df_cb.empty:
                    st.dataframe(df_cb)
                    res = f"Hiện có {len(df_cb)} CBCNV."
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": df_cb})
                    handled = True

            # Tìm trong Hỏi-Trả lời tĩnh
            if not handled:
                ans, sc = semantic_search(u_input, all_data.get("Hỏi-Trả lời", pd.DataFrame()), "Câu hỏi", "Câu trả lời")
                if ans:
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    handled = True

            # --- LUỒNG 2: GOOGLE GEMINI (AI STUDIO) ---
            # Nếu không tìm thấy trong sheet, sẽ hỏi kiến thức bên ngoài
            if not handled and secrets_data["gemini"]:
                try:
                    genai.configure(api_key=secrets_data["gemini"])
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    context = f"Bạn là trợ lý Đội Định Hóa. Hãy trả lời anh Long: {u_input}"
                    response = model.generate_content(context)
                    
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    handled = True
                except:
                    st.error("AI hiện không phản hồi.")

            if not handled:
                st.info("Em chưa có thông tin này trong dữ liệu nội bộ.")