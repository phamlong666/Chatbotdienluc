# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Chatbot Điện lực",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Chatbot Phân tích dữ liệu Điện lực")
st.write("Xin chào! Tôi là trợ lý AI giúp bạn phân tích dữ liệu từ Google Sheets.")

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("📊 Hệ thống Chatbot")
st.sidebar.write("Ứng dụng AI phân tích dữ liệu nội bộ ngành điện lực")

# =========================================
# GOOGLE SHEETS CONNECTION
# =========================================

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

info = dict(st.secrets["gdrive_service_account"])

creds = Credentials.from_service_account_info(
    info,
    scopes=SCOPES
)

client = gspread.authorize(creds)

spreadsheet_id = st.secrets["spreadsheet_id"]

spreadsheet = client.open_by_key(spreadsheet_id)

# =========================================
# LOAD DATA
# =========================================

@st.cache_data
def load_data():

    data = {}

    worksheets = spreadsheet.worksheets()

    for ws in worksheets:

        records = ws.get_all_records()

        if len(records) == 0:
            continue

        df = pd.DataFrame(records)

        data[ws.title] = df

    return data


data = load_data()

# =========================================
# GEMINI CONFIG
# =========================================

genai.configure(api_key=st.secrets["gemini_api_key"])

model = genai.GenerativeModel("gemini-1.5-flash")

# =========================================
# DATA PREVIEW
# =========================================

with st.expander("📑 Xem dữ liệu Google Sheets"):

    if len(data) == 0:
        st.warning("Không tìm thấy dữ liệu trong Google Sheets")

    for name, df in data.items():

        st.subheader(f"Sheet: {name}")
        st.dataframe(df)

# =========================================
# CHAT HISTORY
# =========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================================
# CHAT INPUT
# =========================================

question = st.chat_input("Nhập câu hỏi về dữ liệu điện lực...")

if question:

    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({"role": "user", "content": question})

    # Tạo context từ dữ liệu

    context = ""

    for name, df in data.items():

        context += f"\n===== Sheet: {name} =====\n"

        try:
            context += df.head(20).to_string()
        except:
            pass

    prompt = f"""
Bạn là trợ lý AI phân tích dữ liệu ngành điện lực.

Dưới đây là dữ liệu được trích từ Google Sheets:

{context}

Câu hỏi của người dùng:
{question}

Hãy:
- Phân tích dữ liệu nếu liên quan
- Nếu không có dữ liệu phù hợp, trả lời theo kiến thức chung
- Trả lời bằng tiếng Việt rõ ràng
"""

    try:

        response = model.generate_content(prompt)

        answer = response.text

    except Exception as e:

        answer = f"Lỗi khi gọi AI: {str(e)}"

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
