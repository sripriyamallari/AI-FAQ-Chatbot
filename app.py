import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# FAQ DATA
# -----------------------------
faqs = [
    {
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence (AI) is a technology that enables computers to perform tasks that normally require human intelligence, such as learning, reasoning, and problem solving."
    },
    {
        "question": "What is Machine Learning?",
        "answer": "Machine Learning (ML) is a branch of AI where computers learn patterns from data and use them to make predictions or decisions."
    },
    {
        "question": "What is Deep Learning?",
        "answer": "Deep Learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns from large amounts of data."
    },
    {
        "question": "What is Natural Language Processing?",
        "answer": "Natural Language Processing (NLP) is a field of AI that helps computers understand and process human language."
    },
    {
        "question": "What is Generative AI?",
        "answer": "Generative AI is a type of artificial intelligence that can create new content such as text, images, audio, video, and code."
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a machine learning model made of connected nodes arranged in layers. It learns patterns from data."
    },
    {
        "question": "What is Computer Vision?",
        "answer": "Computer Vision is a field of AI that enables computers to understand and analyze images and videos."
    },
    {
        "question": "What are the applications of AI?",
        "answer": "AI is used in healthcare, education, finance, transportation, robotics, cybersecurity, recommendation systems, and chatbots."
    },
    {
        "question": "What are the benefits of AI?",
        "answer": "AI can automate repetitive tasks, analyze large amounts of data, improve efficiency, and support decision making."
    },
    {
        "question": "What are the limitations of AI?",
        "answer": "AI can depend on data quality, may produce incorrect results, can require significant computing resources, and does not have human common sense."
    },
    {
        "question": "What is an AI chatbot?",
        "answer": "An AI chatbot is a software application that uses artificial intelligence to communicate with users and answer their questions."
    },
    {
        "question": "What is supervised learning?",
        "answer": "Supervised learning is a machine learning method where a model learns from labeled data containing inputs and known outputs."
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Unsupervised learning is a machine learning method where a model finds patterns or groups in data without labeled outputs."
    },
    {
        "question": "What is reinforcement learning?",
        "answer": "Reinforcement learning is a machine learning method where an agent learns by interacting with an environment and receiving rewards or penalties."
    },
    {
        "question": "Will AI replace humans?",
        "answer": "AI can automate some tasks and change how people work, while many tasks still require human creativity, communication, judgment, and responsibility."
    }
]


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Prepare FAQ questions
questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess_text(q) for q in questions]


# -----------------------------
# TF-IDF MODEL
# -----------------------------
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)


# -----------------------------
# FIND BEST ANSWER
# -----------------------------
def find_best_answer(user_question):

    cleaned_question = preprocess_text(user_question)

    if not cleaned_question:
        return "Please enter a question."

    user_vector = vectorizer.transform([cleaned_question])

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_index = similarity_scores.argmax()

    best_score = similarity_scores[best_index]

    # If the question is not related to the FAQs
    if best_score < 0.15:
        return (
            "Sorry, I could not find a suitable answer. "
            "Please ask a question related to Artificial Intelligence."
        )

    return faqs[best_index]["answer"]


# -----------------------------
# STREAMLIT PAGE
# -----------------------------
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# TITLE
# -----------------------------
st.title("🤖 Artificial Intelligence FAQ Chatbot")

st.write(
    "Ask questions about Artificial Intelligence, "
    "Machine Learning, Deep Learning, NLP, Generative AI, "
    "Computer Vision, and more."
)

st.divider()


# -----------------------------
# USER INPUT
# -----------------------------
user_question = st.text_input(
    "Ask your question:",
    placeholder="Example: What is Artificial Intelligence?"
)


# -----------------------------
# BUTTON
# -----------------------------
if st.button("Get Answer", type="primary"):

    if user_question.strip():

        answer = find_best_answer(user_question)

        st.success(answer)

    else:

        st.warning("Please enter a question.")


# -----------------------------
# SAMPLE QUESTIONS
# -----------------------------
st.divider()

st.subheader("💡 Example Questions")

st.write("• What is Artificial Intelligence?")
st.write("• What is Machine Learning?")
st.write("• What is NLP?")
st.write("• What is Generative AI?")
st.write("• What are the applications of AI?")
st.write("• What is Deep Learning?")


# -----------------------------
# FOOTER
# -----------------------------
st.caption(
    "Built with Python, TF-IDF, Cosine Similarity and Streamlit."
)