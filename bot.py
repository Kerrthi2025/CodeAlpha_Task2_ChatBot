from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample FAQ knowledge base for AI and Machine Learning
FAQS = {
    "What is machine learning?": "Machine learning is a subset of AI that focuses on building systems that learn from data to make predictions or decisions.",
    "What is the difference between AI and machine learning?": "AI is a broad field aiming to create intelligent systems, while machine learning is a subset of AI that uses statistical techniques to enable machines to learn from data.",
    "What are the types of machine learning?": "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
    "What is deep learning?": "Deep learning is a subset of machine learning that uses neural networks with many layers to process complex data.",
    "What are neural networks?": "Neural networks are computing systems inspired by the human brain, consisting of layers of nodes that process and learn from data.",
    "What is natural language processing (NLP)?": "NLP is a field of AI that enables machines to understand, interpret, and respond to human language.",
    "What is overfitting in machine learning?": "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new data.",
    "What is the role of data in AI?": "Data is the foundation of AI; it is used to train models, enabling them to learn patterns and make predictions.",
    "How do I get started with AI and machine learning?": "Start by learning programming languages like Python, exploring libraries like TensorFlow and Scikit-learn, and studying foundational concepts in statistics and linear algebra.",
    "What is transfer learning?": "Transfer learning is a technique where a model trained on one task is reused as the starting point for another related task."
}

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Prepare the FAQ data
faq_keys = list(FAQS.keys())
preprocessed_faq_keys = [preprocess_text(key) for key in faq_keys]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer().fit(preprocessed_faq_keys)
faq_vectors = vectorizer.transform(preprocessed_faq_keys)

# Function to find the best matching FAQ
def find_best_match(question):
    question = preprocess_text(question)
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, faq_vectors)
    best_match_index = similarity_scores.argmax()
    return faq_keys[best_match_index], FAQS[faq_keys[best_match_index]]

# Function to handle chatbot response
def chatbot_response(question):
    question, answer = find_best_match(question)
    satisfactory = "not satisfied" not in question.lower()
    return answer, satisfactory

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('question', '')
    answer, satisfactory = chatbot_response(question)
    return jsonify({'answer': answer, 'satisfactory': satisfactory})

if __name__ == '__main__':
    app.run(debug=True)
