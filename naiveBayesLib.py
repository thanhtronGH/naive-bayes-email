import numpy as np

# Preprocess the text: Convert all to lowercase and split into words 
def preprocess(text): 
    return text.lower().split() 


# Create a vocabulary 
def create_vocab(emails):
    return set(word
               for email in emails
               for word in preprocess(email))

# Create a dictionary for word counts in each class 
def create_word_count_matrix(emails, labels, vocab): 
    word_count = {'spam': np.zeros(len(vocab)), 'ham': np.zeros(len(vocab))} 
    word_to_idx = {word: idx 
                   for idx, word in enumerate(vocab)} 
 
    for email, label in zip(emails, labels): 
        word_list = preprocess(email) 
        for word in word_list: 
            if word in word_to_idx: 
                word_count[label][word_to_idx[word]] += 1 
    return word_count, word_to_idx 

# Compute the prior probability for each class 
def compute_class_prior(labels): 
    unique_classes, counts = np.unique(labels, return_counts=True) 
    prior = dict(zip(unique_classes, counts / len(labels))) 
    return prior 

# Compute likelihoods (conditional probabilities) 
def compute_likelihood(word_count): 
    likelihood = {'spam': word_count['spam'] + 1, 'ham': word_count['ham'] + 1} 
    # Apply Laplace smoothing (add 1 to each count) 
    likelihood['spam'] /= np.sum(likelihood['spam']) 
    likelihood['ham'] /= np.sum(likelihood['ham']) 
    return likelihood 

# Train with realtime updated mails
def train(emails, labels):

    vocab = create_vocab(emails)
    word_count, word_to_idx = create_word_count_matrix(
        emails,
        labels,
        vocab
    )
    prior = compute_class_prior(labels)

    likelihood = compute_likelihood(word_count)

    return vocab, prior, likelihood, word_to_idx


# =========================
# PREDICT
# =========================

def predict(email, prior, likelihood, word_to_idx):
    
    email_words = preprocess(email)
    log_prob_spam = np.log(prior["spam"])
    log_prob_ham = np.log(prior["ham"])

    for word in email_words: 
        if word in word_to_idx: 
            idx = word_to_idx[word] 
            log_prob_spam += np.log(likelihood['spam'][idx]) 
            log_prob_ham += np.log(likelihood['ham'][idx]) 

    if log_prob_spam > log_prob_ham:
        return "spam"
    return "ham"
