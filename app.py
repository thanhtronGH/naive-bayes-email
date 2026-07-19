import streamlit as st
from naiveBayesLib import *

# ===========================
# INITIAL DATA
# ===========================

if "emails" not in st.session_state:

    st.session_state.emails = [
        'Buy cheap medicine now!',
        'Hey, how are you doing today?',
        'Congratulations, you have won a free ticket!',
        'Meeting tomorrow at 10am.',
        'Limited offer, buy one get one free!',
        'Can we schedule a call for next week?',
        'Win a free iPhone, click here!',
        'Are you coming to the party tonight?',
        'Big opportunity right now!',
        'We will discussion tommorow on the opportunity of cooperation',
        'This is your exercise today',
        'Get your degree for free',
        'Exam result',
        'Your ticket for Newyork.',
        'Your flight reservation'
    ]

    st.session_state.labels = [
        'spam','ham','spam','ham','spam',
        'ham','spam','ham','spam','ham',
        'ham','spam','ham','ham','ham'
    ]

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


# ===========================
# UI
# ===========================

st.title("Spam / Ham Detection")

st.sidebar.header("Statistics")

st.sidebar.write(
    "Emails :", len(st.session_state.emails)
)

st.sidebar.write(
    "Vocabulary :", len(st.session_state.vocab)
)

st.sidebar.write(
    "Spam :", st.session_state.labels.count("spam")
)

st.sidebar.write(
    "Ham :", st.session_state.labels.count("ham")
)

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

            st.session_state.emails.append(
                st.session_state.email
            )

            st.session_state.labels.append("spam")

            train_model()

            st.success("Model Updated!")

    with col2:

        if st.button("Correct to Ham"):

            st.session_state.emails.append(
                st.session_state.email
            )

            st.session_state.labels.append("ham")

            train_model()

            st.success("Model Updated!")


st.divider()

st.subheader("Training Dataset")

for i, (e, l) in enumerate(
        zip(st.session_state.emails,
            st.session_state.labels)):

    st.write(f"{i+1}. [{l.upper()}] {e}")
