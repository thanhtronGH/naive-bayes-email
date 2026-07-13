import streamlit as st
import numpy as np

# =====================================================================
# 1. CÁC HÀM TIỀN XỬ LÝ VÀ THUẬT TOÁN (Giữ nguyên logic từ Yêu cầu 1 & 2)
# =====================================================================
def preprocess(text):
    return text.lower().split()

def update_model_and_vocab():
    """
    Hàm này tự động tính toán lại Từ điển (Vocabulary), Xác suất tiên nghiệm (Prior),
    và Xác suất điều kiện (Likelihood) dựa trên danh sách dữ liệu hiện tại trong session_state.
    """
    emails = st.session_state.emails
    labels = st.session_state.labels
    
    # Cập nhật lại Từ điển (Yêu cầu 2)
    vocab = set(word for email in emails for word in preprocess(email))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Khởi tạo ma trận đếm số lần xuất hiện của từ
    word_count = {
        'spam': np.zeros(len(vocab)), 
        'ham': np.zeros(len(vocab))
    }
    
    for email, label in zip(emails, labels):
        word_list = preprocess(email)
        for word in word_list:
            if word in word_to_idx:
                word_count[label][word_to_idx[word]] += 1
                
    # Tính xác suất tiên nghiệm P(Spam) và P(Ham)
    unique_classes, counts = np.unique(labels, return_counts=True)
    prior = dict(zip(unique_classes, counts / len(labels)))
    
    # Tính xác suất điều kiện P(Word | Class) + Laplace smoothing
    likelihood = {
        'spam': word_count['spam'] + 1, 
        'ham': word_count['ham'] + 1
    }
    likelihood['spam'] /= np.sum(likelihood['spam'])
    likelihood['ham'] /= np.sum(likelihood['ham'])
    
    # Lưu lại toàn bộ mô hình và từ điển vào session_state
    st.session_state.vocab = vocab
    st.session_state.word_to_idx = word_to_idx
    st.session_state.prior = prior
    st.session_state.likelihood = likelihood

def predict(email):
    """Hàm dự đoán sử dụng dữ liệu mô hình hiện tại trong session_state"""
    prior = st.session_state.prior
    likelihood = st.session_state.likelihood
    word_to_idx = {w: i for i, w in enumerate(st.session_state.vocab)} # Lấy từ điển hiện tại
    
    email_words = preprocess(email)
    log_prob_spam = np.log(prior.get('spam', 0.5))
    log_prob_ham = np.log(prior.get('ham', 0.5))

    for word in email_words:
        if word in word_to_idx:
            idx = word_to_idx[word]
            log_prob_spam += np.log(likelihood['spam'][idx])
            log_prob_ham += np.log(likelihood['ham'][idx])
        else:
            # Nếu gặp từ hoàn toàn mới (chưa có trong từ điển), dùng xác suất mặc định rất nhỏ để tránh lỗi
            log_prob_spam += np.log(1 / (np.sum(likelihood['spam']) + len(st.session_state.vocab)))
            log_prob_ham += np.log(1 / (np.sum(likelihood['ham']) + len(st.session_state.vocab)))

    return 'spam' if log_prob_spam > log_prob_ham else 'ham'

# =====================================================================
# 2. KHỞI TẠO DỮ LIỆU BAN ĐẦU (Chỉ chạy 1 lần duy nhất khi mở app)
# =====================================================================
if 'emails' not in st.session_state:
    # Tập dữ liệu huấn luyện ban đầu của bạn
    st.session_state.emails = [
        'Buy cheap medicine now!', 'Hey, how are you doing today?', 
        'Congratulations, you have won a free ticket!', 'Meeting tomorrow at 10am.', 
        'Limited offer, buy one get one free!', 'Can we schedule a call for next week?', 
        'Win a free iPhone, click here!', 'Are you coming to the party tonight?', 
        'Big opportunity right now!', 'We will discussion tommorow on the opportunity of cooperation', 
        'This is your exercise today', 'Get your degree for free', 'Exam result', 
        'Your ticket for Newyork.', 'Your flight reservation'
    ]
    st.session_state.labels = [
        'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 
        'ham', 'ham', 'spam', 'ham', 'ham', 'ham'
    ]
    # Tạo mô hình lần đầu tiên
    update_model_and_vocab()

# =====================================================================
# 3. GIAO DIỆN STREAMLIT (UI)
# =====================================================================
st.title("Ứng dụng Phân loại Email Spam/Ham - Naive Bayes")
st.subheader("Kiểm thử thời gian thực & Cập nhật mô hình tự động")

# Hiển thị thông tin tổng quan về mô hình hiện tại ở thanh bên (Sidebar)
st.sidebar.header("Thống kê Mô hình Hiện tại")
st.sidebar.write(f"📊 Tổng số mẫu email: **{len(st.session_state.emails)}**")
st.sidebar.write(f"🔑 Số từ trong từ điển (Vocabulary): **{len(st.session_state.vocab)}**")
st.sidebar.write(f"📩 Số lượng Spam: **{st.session_state.labels.count('spam')}**")
st.sidebar.write(f"✉️ Số lượng Ham: **{st.session_state.labels.count('ham')}**")

# KHU VỰC 1: KIỂM THỬ THỜI GIAN THỰC
st.write("---")
st.header("🔍 Kiểm thử Email")
user_input = st.text_area("Nhập nội dung email cần kiểm tra tại đây:", placeholder="Ví dụ: Get your free prize today...")

if user_input:
    # Tiến hành dự đoán ngay khi người dùng nhập chữ
    prediction = predict(user_input)
    
    # Hiển thị kết quả bằng màu sắc trực quan
    if prediction == 'spam':
        st.error(f"🚨 Kết quả dự đoán từ thuật toán: **SPAM (Thư rác)**")
    else:
        st.success(f"✅ Kết quả dự đoán từ thuật toán: **HAM (Thư thường)**")
        
    # KHU VỰC 2: NGƯỜI DÙNG CAN THIỆP KHI ĐOÁN SAI
    st.write("👉 *Thuật toán đoán sai? Hãy bấm nút dưới đây để sửa lại và cập nhật mô hình:*")
    
    # Xác định nhãn ngược lại để làm nút bấm sửa sai
    correct_label = 'ham' if prediction == 'spam' else 'spam'
    button_label = f"Sửa lại thành: Thư thường (Ham)" if correct_label == 'ham' else f"Sửa lại thành: Thư rác (Spam)"
    
    if st.button(button_label, type="primary"):
        # 1. Thêm email hiện tại cùng nhãn đúng vào tập dữ liệu trong bộ nhớ
        st.session_state.emails.append(user_input)
        st.session_state.labels.append(correct_label)
        
        # 2. Chạy hàm tính toán lại toàn bộ từ điển và xác suất
        update_model_and_vocab()
        
        # 3. Thông báo thành công và yêu cầu app load lại để áp dụng kết quả mới
        st.balloons()
        st.info("🔄 Đã cập nhật email này vào bộ dữ liệu huấn luyện và tính toán lại từ điển thành công!")
        st.rerun()

# KHU VỰC 3: XEM BỘ DỮ LIỆU HIỆN TẠI (Tùy chọn - Giúp giảng viên dễ chấm điểm)
with st.expander("📂 Xem danh sách toàn bộ email và nhãn hiện tại trong hệ thống"):
    for idx, (em, lb) in enumerate(zip(st.session_state.emails, st.session_state.labels)):
        st.text(f"[{idx+1}] [{lb.upper()}] {em}")