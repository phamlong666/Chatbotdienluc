# -*- coding: utf-8 -*-

"""
Chatbot AI + Google Sheets (PRO Version)
- Đọc dữ liệu từ Google Sheets bằng Service Account
- AI hiểu câu hỏi
- Phân tích dữ liệu nếu có trong sheet
- Nếu không có dữ liệu → AI trả lời kiến thức chung

Yêu cầu secrets trong Streamlit:

[gdrive_service_account]
...

spreadsheet_url = "https://docs.google.com/..."
openai_api_key = "sk-..."
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import gspread

from google.oauth2.service_account import Credentials
from openai import OpenAI

# ======================================================
# CẤU HÌNH STREAMLIT
# ======================================================

st.set_page_config(
    page_title="Chatbot AI Điện lực",
    layout="wide"
)

st.title("⚡ Chatbot AI Phân tích dữ liệu Điện lực")

# ======================================================
# KẾT NỐI GOOGLE SHEETS
# ======================================================

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

info = dict(st.secrets["gdrive_service_account"])

creds = Credentials.from_service_account_info(
    info,
    scopes=SCOPES
)

client = gspread.authorize(creds)

spreadsheet = client.open_by_url(
    st.secrets["spreadsheet_url"]
)

# ======================================================
# LOAD TẤT CẢ SHEET
# ======================================================

@st.cache_data
def load_all_sheets():

    data = {}

    worksheets = spreadsheet.worksheets()

    for ws in worksheets:

        df = pd.DataFrame(ws.get_all_records())

        data[ws.title] = df

    return data


data = load_all_sheets()

# ======================================================
# OPENAI CLIENT
# ======================================================

client_ai = OpenAI(
    api_key=st.secrets["openai_api_key"]
)

# ======================================================
# HIỂN THỊ DATA (DEBUG)
# ======================================================

with st.expander("📊 Dữ liệu Google Sheets"):

    for name, df in data.items():

        st.subheader(name)

        if df.empty:
            st.info("Sheet không có dữ liệu")
        else:
            st.dataframe(df)

# ======================================================
# HÀM TẠO CONTEXT CHO AI
# ======================================================

def build_context():

    context = ""

    for name, df in data.items():

        context += f"\nSheet: {name}\n"

        if df.empty:
            context += "Không có dữ liệu\n"
        else:
            context += df.head(30).to_string()
            context += "\n"

    return context

# ======================================================
# AI PHÂN TÍCH
# ======================================================

def ai_analyze(question):

    context = build_context()

    prompt = f"""
Bạn là chuyên gia phân tích dữ liệu ngành điện lực.

Dưới đây là dữ liệu Google Sheets:

{context}

Câu hỏi của người dùng:
{question}

Hãy làm theo các bước:

1. Nếu dữ liệu có trong bảng → phân tích dữ liệu.
2. Nếu cần → đề xuất biểu đồ phù hợp.
3. Nếu dữ liệu không có → trả lời theo kiến thức chung.
4. Trả lời bằng tiếng Việt.
"""

    response = client_ai.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {
                "role": "system",
                "content": "Bạn là chuyên gia phân tích dữ liệu."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content

# ======================================================
# VẼ BIỂU ĐỒ ĐƠN GIẢN
# ======================================================

def draw_chart(df, x, y):

    if x not in df.columns or y not in df.columns:
        st.warning("Không tìm thấy cột để vẽ biểu đồ")
        return

    fig = plt.figure()

    df.plot(kind="bar", x=x, y=y)

    st.pyplot(fig)

# ======================================================
# CHATBOT UI
# ======================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# Hiển thị lịch sử

for msg in st.session_state.messages:

    st.chat_message(msg["role"]).write(msg["content"])

# Ô nhập chat

question = st.chat_input("Nhập câu hỏi...")

if question:

    st.chat_message("user").write(question)

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.spinner("AI đang phân tích dữ liệu..."):

        answer = ai_analyze(question)

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption("Chatbot AI đọc dữ liệu trực tiếp từ Google Sheets và phân tích bằng AI")
