import re
import nltk
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK tokenizer
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# -----------------------------
# AI FAQ Dataset
# -----------------------------

faqs = [
    {
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence (AI) is a branch of computer science that enables machines to perform tasks that normally require human intelligence, such as learning, reasoning, problem-solving, and decision-making."
    },
    {
        "question": "What is Machine Learning?",
        "answer": "Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every task."
    },
    {
        "question": "What is Deep Learning?",
        "answer": "Deep Learning is a subset of Machine Learning that uses multi-layered neural networks to learn complex patterns from large amounts of data."
    },
    {
        "question": "What is Natural Language Processing?",
        "answer": "Natural Language Processing (NLP) is a field of Artificial Intelligence that enables computers to understand, process, and generate human language."
    },
    {
        "question": "What is Generative AI?",
        "answer": "Generative AI is a type of Artificial Intelligence that can create new content such as text, images, audio, video, and computer code."
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a machine learning model inspired by the human brain. It contains interconnected nodes that learn patterns from data."
    },
    {
        "question": "What is Computer Vision?",
        "answer": "Computer Vision is a field of Artificial Intelligence that enables computers to understand and analyze images and videos."
    },
    {
        "question": "What are the applications of Artificial Intelligence?",
        "answer": "AI is used in healthcare, education, finance, transportation, cybersecurity, robotics, customer service, recommendation systems, and many other fields."
    },
    {
        "question": "What are the benefits of AI?",
        "answer": "AI can automate repetitive tasks, analyze large amounts of data, improve efficiency, support decision-making, and provide personalized services."
    },
    {
        "question": "What are the limitations of AI?",
        "answer": "AI can depend on the quality of its training data, may produce incorrect results, can be difficult to interpret, and may require significant computing resources."
    },
    {
        "question": "What is an AI chatbot?",
        "answer": "An AI chatbot is a software application that uses Artificial Intelligence to communicate with users and provide answers to their questions."
    },
    {
        "question": "What is supervised learning?",
        "answer": "Supervised learning is a machine learning method where a model learns from labeled data and uses that learning to make predictions on new data."
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Unsupervised learning is a machine learning method where a model discovers patterns or structures in data without labeled answers."
    },
    {
        "question": "What is reinforcement learning?",
        "answer": "Reinforcement learning is a machine learning technique where an agent learns by interacting with an environment and receiving rewards or penalties."
    },
    {
        "question": "Will AI replace humans?",
        "answer": "AI can automate many tasks, but humans remain important for judgment, creativity, communication, responsibility, and many other activities."
    }
]


# -----------------------------
# Text Preprocessing
# -----------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)


# -----------------------------
# TF-IDF
# -----------------------------

questions = [item["question"] for item in faqs]

processed_questions = [
    preprocess_text(question)
    for question in questions
]

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)


# -----------------------------
# Find Best Answer
# -----------------------------

def find_best_answer(user_question):

    if not user_question.strip():
        return "Please enter a question."

    cleaned_question = preprocess_text(user_question)

    user_vector = vectorizer.transform([cleaned_question])

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_index = similarity_scores.argmax()

    best_score = similarity_scores[best_index]

    if best_score < 0.15:
        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please ask a question related to Artificial Intelligence."
        )

    return faqs[best_index]["answer"]


# -----------------------------
# Gradio Chatbot
# -----------------------------

def chatbot_response(message, history):
    return find_best_answer(message)


demo = gr.ChatInterface(
    fn=chatbot_response,
    title="🤖 Artificial Intelligence FAQ Chatbot",
    description=(
        "Ask questions about Artificial Intelligence, "
        "Machine Learning, Deep Learning, NLP, "
        "Generative AI, and more."
    ),
    textbox=gr.Textbox(
        placeholder="Ask your AI question...",
        label="Your Question"
    )
)


# -----------------------------
# Launch Application
# -----------------------------

if __name__ == "__main__":
    demo.launch()
