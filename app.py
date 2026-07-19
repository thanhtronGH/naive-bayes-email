import streamlit as st
from naiveBayesLib import *
from streamlit_gsheets import GSheetsConnection
import pandas as pd
# Connect to google sheet
conn = st.connection("gsheets", type=GSheetsConnection)

if "emails" not in st.session_state:

    df = conn.read(ttl=0) # ttl=0 để luôn lấy dữ liệu mới nhất, không lưu cache
    
    st.session_state.emails = df["emails"].tolist()
    st.session_state.labels = df["labels"].tolist()

def train_model():

    vocab, prior, likelihood, word_to_idx = train(
        st.session_state.emails,
        st.session_state.labels
    )

    st.session_state.vocab = vocab
    st.session_state.word_to_idx = word_to_idx
    st.session_state.prior = prior
    st.session_state.likelihood = likelihood


if "prior" not in st.session_state:
    train_model()

# UI
st.title("Spam / Ham Detection")
st.sidebar.header("Statistics")
st.sidebar.write("Emails :", len(st.session_state.emails))
st.sidebar.write("Vocabulary :", len(st.session_state.vocab))
st.sidebar.write("Spam :", st.session_state.labels.count("spam"))
st.sidebar.write("Ham :", st.session_state.labels.count("ham"))
st.divider()

email = st.text_area(
    "Input Email"
)

if st.button("Predict"):
    result = predict(
        email,
        st.session_state.prior,
        st.session_state.likelihood,
        st.session_state.word_to_idx
    )
    st.session_state.result = result
    st.session_state.email = email

# Result
if "result" in st.session_state:
    if st.session_state.result == "spam":
        st.error("Prediction : SPAM")
    else:
        st.success("Prediction : HAM")

    st.write("Nếu thuật toán dự đoán sai, hãy sửa lại.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Correct to Spam"):
            st.session_state.emails.append(st.session_state.email)
            st.session_state.labels.append("spam")
            # Generate new data frame
            new_df = pd.DataFrame({
                "emails": st.session_state.emails, 
                "labels": st.session_state.labels
            })
            conn.update(data=new_df) # Post data to gg sheet
            
            train_model()
            st.success("Model Updated!")

    with col2:
        if st.button("Correct to Ham"):
            st.session_state.emails.append(st.session_state.email)
            st.session_state.labels.append("ham")
            # Generate new data frame
            new_df = pd.DataFrame({
                "emails": st.session_state.emails, 
                "labels": st.session_state.labels
            })
            conn.update(data=new_df) # Post data to gg sheet
            train_model()
            st.success("Model Updated!")

st.divider()
st.subheader("Training Dataset")

for i, (e, l) in enumerate(
        zip(st.session_state.emails,
            st.session_state.labels)):

    st.write(f"{i+1}. [{l.upper()}] {e}")
