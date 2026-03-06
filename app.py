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

# CSS Tùy chỉnh để giao diện chia 2 bên rõ ràng (Cập nhật chuẩn Streamlit mới)
st.markdown("""
    <style>
    /* Cấu trúc chung của tin nhắn */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex !important;
        width: fit-content !important;
        max-width: 85% !important;
    }

    /* TIN NHẮN NGƯỜI DÙNG: Đẩy sang phải, màu xanh nhẹ */
    [data-testid="stChatMessageUser"] {
        background-color: #e3f2fd !important;
        margin-left: auto !important;
        flex-direction: row-reverse !important;
        text-align: right;
    }

    /* TIN NHẮN TRỢ LÝ: Nằm bên trái, màu xám nhẹ */
    [data-testid="stChatMessageAssistant"] {
        background-color: #f5f5f5 !important;
        margin-right: auto !important;
    }

    /* Chỉnh sửa avatar để không bị ngược khi dùng row-reverse */
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageAvatar"] {
        margin-left: 10px;
        margin-right: 0;
    }
    
    /* Input chat */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Cấu hình tiếng Việt cho biểu đồ
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# 2. GIẢI MÃ KHÓA BẢO MẬT
@st.cache_resource
def get_decrypted_all_keys():
    config = {"gemini": None, "gdrive": None}
    if "gdrive_service_account" in st.secrets:
        try:
            sec = st.secrets["gdrive_service_account"]
            master_key = sec.get("encryption_key_for_decryption").encode()
            cipher = Fernet(master_key)
            
            # Giải mã API Key Gemini
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
        data = {}
        for ws in sh.worksheets():
            df = pd.DataFrame(ws.get_all_records())
            if not df.empty:
                data[ws.title] = df
        return data
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu từ Sheets: {e}")
        return {}

def semantic_search(query, df, q_col, a_col, threshold=0.3): 
    if df is None or df.empty or q_col not in df.columns: return None, 0
    questions = df[q_col].astype(str).tolist()
    doc_embs = ai_tools["semantic_model"].encode(questions, convert_to_tensor=True)
    query_emb = ai_tools["semantic_model"].encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, doc_embs)[0]
    best_idx = torch.argmax(cos_scores).item()
    score = float(cos_scores[best_idx])
    if score > threshold:
        return df.iloc[best_idx][a_col], score * 100
    return None, 0

# 6. QUẢN LÝ HỘI THOẠI
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_text" not in st.session_state:
    st.session_state.voice_text = None

# SIDEBAR
with st.sidebar:
    st.image("https://raw.githubusercontent.com/phamlong666/Chatbot/main/logo_hinh_tron.png", width=120)
    st.title("Trợ lý Đội Định Hóa")
    st.divider()
    
    if st.button("🗑 Xóa lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
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

# HIỂN THỊ CHAT
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg: st.dataframe(msg["df"])

# NHẬN ĐẦU VÀO
u_input = st.session_state.voice_text if st.session_state.voice_text else st.chat_input("Hỏi em về KPI, CBCNV hoặc kiến thức ngành điện...")
st.session_state.voice_text = None

if u_input:
    # 1. Lưu tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": u_input})
    with st.chat_message("user"):
        st.markdown(u_input)

    # 2. Phản hồi của Trợ lý
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý dữ liệu..."):
            all_data = load_all_sheets()
            handled = False
            norm_u = normalize_text(u_input)

            # --- LỚP 1: DỮ LIỆU CẤU TRÚC (TABLES) ---
            # KPI
            if any(k in norm_u for k in ["kpi", "chỉ số", "thực hiện"]):
                df_kpi = all_data.get("KPI", pd.DataFrame())
                if not df_kpi.empty:
                    res = "Dạ, đây là thông tin KPI anh cần ạ:"
                    st.markdown(res)
                    st.dataframe(df_kpi)
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": df_kpi})
                    handled = True

            # CBCNV
            if not handled and any(k in norm_u for k in ["cbcnv", "nhân viên", "nhân sự", "con người"]):
                df_cb = all_data.get("CBCNV", pd.DataFrame())
                if not df_cb.empty:
                    res = f"Hiện đội mình đang có {len(df_cb)} CBCNV. Danh sách chi tiết đây ạ:"
                    st.markdown(res)
                    st.dataframe(df_cb)
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": df_cb})
                    handled = True

            # Lãnh đạo xã / Địa phương
            if not handled and any(k in norm_u for k in ["lãnh đạo", "xã", "địa phương"]):
                df_ld = all_data.get("Lãnh đạo xã", pd.DataFrame())
                if not df_ld.empty:
                    res = "Thông tin lãnh đạo địa phương anh cần đây ạ:"
                    st.markdown(res)
                    st.dataframe(df_ld)
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": df_ld})
                    handled = True

            # --- LỚP 2: HỎI ĐÁP THÔNG MINH (SEMANTIC SEARCH) ---
            if not handled:
                ans, sc = semantic_search(u_input, all_data.get("Hỏi-Trả lời"), "Câu hỏi", "Câu trả lời")
                if ans:
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    handled = True

            # --- LỚP 3: GOOGLE GEMINI (KIẾN THỨC NGOÀI) ---
            if not handled and secrets_data["gemini"]:
                try:
                    genai.configure(api_key=secrets_data["gemini"])
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Tạo bối cảnh cho AI
                    prompt_final = f"Bạn là trợ lý ảo Đội Định Hóa. Anh Long đang hỏi: '{u_input}'. Hãy trả lời thân thiện, chuyên nghiệp."
                    response = model.generate_content(prompt_final)
                    
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    handled = True
                except:
                    st.warning("AI hiện đang bận, anh thử lại sau nhé!")

            # --- LỖI CUỐI CÙNG ---
            if not handled:
                err_msg = "Em chưa tìm thấy thông tin này trong dữ liệu nội bộ và AI cũng không thể xác định. Anh hãy thử hỏi bằng cách khác nhé!"
                st.info(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})