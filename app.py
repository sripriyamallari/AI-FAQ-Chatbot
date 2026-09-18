{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Z9u46AzIBRL0"
      },
      "outputs": [],
      "source": [
        "!pip install -q gradio nltk scikit-learn"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "import nltk\n",
        "import gradio as gr\n",
        "\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "\n",
        "# Download NLTK resources\n",
        "nltk.download(\"punkt\")\n",
        "nltk.download(\"punkt_tab\")\n",
        "\n",
        "print(\"Libraries loaded successfully!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EREXqw9jBkwr",
        "outputId": "3302ec21-91f5-40d3-eda1-6fac8f3ad8cf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt.zip.\n",
            "[nltk_data] Downloading package punkt_tab to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt_tab.zip.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Libraries loaded successfully!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "faqs = [\n",
        "    {\n",
        "        \"question\": \"What is Artificial Intelligence?\",\n",
        "        \"answer\": \"Artificial Intelligence (AI) is a branch of computer science that enables machines to perform tasks that normally require human intelligence, such as learning, reasoning, problem-solving, and decision-making.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is Machine Learning?\",\n",
        "        \"answer\": \"Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every task.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is Deep Learning?\",\n",
        "        \"answer\": \"Deep Learning is a subset of Machine Learning that uses multi-layered neural networks to learn complex patterns from large amounts of data.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is Natural Language Processing?\",\n",
        "        \"answer\": \"Natural Language Processing, or NLP, is a field of Artificial Intelligence that enables computers to understand, process, and generate human language.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is Generative AI?\",\n",
        "        \"answer\": \"Generative AI is a type of Artificial Intelligence that can create new content such as text, images, audio, video, and computer code.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is a neural network?\",\n",
        "        \"answer\": \"A neural network is a machine learning model inspired by the human brain. It contains interconnected nodes that learn patterns from data.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is Computer Vision?\",\n",
        "        \"answer\": \"Computer Vision is a field of Artificial Intelligence that enables computers to understand and analyze images and videos.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What are the applications of Artificial Intelligence?\",\n",
        "        \"answer\": \"AI is used in healthcare, education, finance, transportation, cybersecurity, robotics, customer service, recommendation systems, and many other fields.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What are the benefits of AI?\",\n",
        "        \"answer\": \"AI can automate repetitive tasks, analyze large amounts of data, improve efficiency, support decision-making, and provide personalized services.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What are the limitations of AI?\",\n",
        "        \"answer\": \"AI can depend on the quality of its training data, may produce incorrect results, can be difficult to interpret, and may require significant computing resources.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is an AI chatbot?\",\n",
        "        \"answer\": \"An AI chatbot is a software application that uses Artificial Intelligence to communicate with users and provide answers to their questions.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is supervised learning?\",\n",
        "        \"answer\": \"Supervised learning is a machine learning method where a model learns from labeled data and uses that learning to make predictions on new data.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is unsupervised learning?\",\n",
        "        \"answer\": \"Unsupervised learning is a machine learning method where a model discovers patterns or structures in data without labeled answers.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"What is reinforcement learning?\",\n",
        "        \"answer\": \"Reinforcement learning is a machine learning technique where an agent learns by interacting with an environment and receiving rewards or penalties.\"\n",
        "    },\n",
        "    {\n",
        "        \"question\": \"Will AI replace humans?\",\n",
        "        \"answer\": \"AI can automate many tasks, but humans remain important for judgment, creativity, communication, responsibility, and many other activities.\"\n",
        "    }\n",
        "]\n",
        "\n",
        "print(\"Total FAQs:\", len(faqs))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nuzMcWsLCM0U",
        "outputId": "6f92155e-7ebc-4cba-818d-326112ff0af3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total FAQs: 15\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def preprocess_text(text):\n",
        "    # Convert to lowercase\n",
        "    text = text.lower()\n",
        "\n",
        "    # Remove special characters\n",
        "    text = re.sub(r\"[^a-zA-Z0-9\\s]\", \"\", text)\n",
        "\n",
        "    # Tokenize using NLTK\n",
        "    tokens = nltk.word_tokenize(text)\n",
        "\n",
        "    # Keep useful words\n",
        "    tokens = [word for word in tokens if word.strip()]\n",
        "\n",
        "    # Join tokens again\n",
        "    return \" \".join(tokens)\n",
        "\n",
        "\n",
        "questions = [item[\"question\"] for item in faqs]\n",
        "\n",
        "processed_questions = [\n",
        "    preprocess_text(question)\n",
        "    for question in questions\n",
        "]\n",
        "\n",
        "print(\"Original:\", questions[0])\n",
        "print(\"Processed:\", processed_questions[0])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kATCYhrrCU3E",
        "outputId": "9e563622-8242-4678-d2f6-f52c20bcb827"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Original: What is Artificial Intelligence?\n",
            "Processed: what is artificial intelligence\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "vectorizer = TfidfVectorizer()\n",
        "\n",
        "faq_vectors = vectorizer.fit_transform(processed_questions)\n",
        "\n",
        "print(\"TF-IDF completed!\")\n",
        "print(\"Vector shape:\", faq_vectors.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XXuHzM_wCeO8",
        "outputId": "0ef3bf5f-0489-4915-de82-3cfb776235a6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "TF-IDF completed!\n",
            "Vector shape: (15, 30)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def find_best_answer(user_question):\n",
        "    # Clean the user's question\n",
        "    cleaned_question = preprocess_text(user_question)\n",
        "\n",
        "    # Convert user's question to TF-IDF\n",
        "    user_vector = vectorizer.transform([cleaned_question])\n",
        "\n",
        "    # Calculate cosine similarity\n",
        "    similarity_scores = cosine_similarity(\n",
        "        user_vector,\n",
        "        faq_vectors\n",
        "    )[0]\n",
        "\n",
        "    # Find the highest similarity\n",
        "    best_index = similarity_scores.argmax()\n",
        "\n",
        "    best_score = similarity_scores[best_index]\n",
        "\n",
        "    # Minimum similarity threshold\n",
        "    if best_score < 0.15:\n",
        "        return (\n",
        "            \"Sorry, I could not find a suitable answer. \"\n",
        "            \"Please ask a question related to Artificial Intelligence.\"\n",
        "        )\n",
        "\n",
        "    return faqs[best_index][\"answer\"]"
      ],
      "metadata": {
        "id": "c0E3sUubCieM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_questions = [\n",
        "    \"What is AI?\",\n",
        "    \"Tell me about machine learning\",\n",
        "    \"What does NLP mean?\",\n",
        "    \"Explain deep learning\",\n",
        "    \"What is generative artificial intelligence?\"\n",
        "]\n",
        "\n",
        "for question in test_questions:\n",
        "    print(\"User:\", question)\n",
        "    print(\"Bot:\", find_best_answer(question))\n",
        "    print(\"-\" * 70)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "91FF-eXACnLk",
        "outputId": "7aaad0af-ddd8-4d8e-afc8-0bcbf0044749"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "User: What is AI?\n",
            "Bot: Generative AI is a type of Artificial Intelligence that can create new content such as text, images, audio, video, and computer code.\n",
            "----------------------------------------------------------------------\n",
            "User: Tell me about machine learning\n",
            "Bot: Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every task.\n",
            "----------------------------------------------------------------------\n",
            "User: What does NLP mean?\n",
            "Bot: Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every task.\n",
            "----------------------------------------------------------------------\n",
            "User: Explain deep learning\n",
            "Bot: Deep Learning is a subset of Machine Learning that uses multi-layered neural networks to learn complex patterns from large amounts of data.\n",
            "----------------------------------------------------------------------\n",
            "User: What is generative artificial intelligence?\n",
            "Bot: Artificial Intelligence (AI) is a branch of computer science that enables machines to perform tasks that normally require human intelligence, such as learning, reasoning, problem-solving, and decision-making.\n",
            "----------------------------------------------------------------------\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def chatbot_response(message, history):\n",
        "    if not message.strip():\n",
        "        return \"Please enter a question.\"\n",
        "\n",
        "    return find_best_answer(message)\n",
        "\n",
        "\n",
        "demo = gr.ChatInterface(\n",
        "    fn=chatbot_response,\n",
        "    title=\"🤖 Artificial Intelligence FAQ Chatbot\",\n",
        "    description=(\n",
        "        \"Ask questions about Artificial Intelligence, \"\n",
        "        \"Machine Learning, Deep Learning, NLP, Generative AI, and more.\"\n",
        "    ),\n",
        "    textbox=gr.Textbox(\n",
        "        placeholder=\"Ask your AI question...\",\n",
        "        label=\"Your Question\"\n",
        "    )\n",
        ")\n",
        "\n",
        "demo.launch(share=True)"
      ],
      "metadata": {
        "id": "S4gf_SOvDAbk",
        "outputId": "0384459a-8ceb-407f-98ea-1e9742ae129b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 611
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "* Running on public URL: https://ef762003c50aeb6862.gradio.live\n",
            "\n",
            "This share link is temporary and will last for up to 1 week (best effort). For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://ef762003c50aeb6862.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    }
  ]
}
