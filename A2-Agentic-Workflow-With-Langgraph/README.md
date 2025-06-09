# LangGraph-powered QA Assistant

This project is an intelligent Question Answering (QA) system built using **LangGraph**, **Streamlit**, and **LangChain** components. It classifies user queries and routes them through different processing pipelines (LLM, RAG, or Web Search) with built-in validation and retry mechanisms.

---

## 🔧 Features

- **Supervisor Node** to classify user queries into:
  - Mental Health (RAG)
  - Real-time Information (Web Search)
  - General Queries (LLM)
- **Router Function** to route queries based on classification.
- **RAG Node** for mental health-related queries using local PDFs.
- **Web Node** for live information using Tavily Search.
- **LLM Node** for general knowledge using Gemini 1.5.
- **Validation Node** to check answer quality.
- **Retry Mechanism** to reroute if the output is insufficient.
- **Streamlit UI** for interactive user experience.

---

## 🧱 Architecture

- mermaid
graph TD
    A[Supervisor Node] -->|Mental Health| B[RAG Node]
    A -->|Real Time Info| C[Web Search Node]
    A -->|Not Related| D[LLM Node]
    B --> E[Validator]
    C --> E
    D --> E
    E -->|Retry| A
    E -->|Valid| F[Final Node]

## ✅ Requirements
- Python 3.9+

- Streamlit

- LangChain

- LangGraph

- dotenv

- Tavily API Key (for web search)

- Google Generative AI access (for Gemini)

- HuggingFace Transformers


## 📦 Installation

- Clone the repo:
  - git clone https://github.com/yourusername/langgraph-qa-assistant.git
  - cd langgraph-qa-assistant

- Create virtual environment:
  - python -m venv venv
  - source venv/bin/activate   # On Windows: venv\Scripts\activate

- Install dependencies:
  - pip install -r requirements.txt

- Setup environment variables:
  - Create a .env file and add:
    - GOOGLE_API_KEY=your_google_api_key
    - TAVILY_API_KEY=your_tavily_api_key

## ▶️ Running the App
- streamlit run app.py
Then open your browser to http://localhost:8501.

## 📚 How It Works
- User enters a question.

- Supervisor Node classifies the query.

- Depending on the classification:

- RAG Node pulls data from local PDFs.

- Web Node uses Tavily for real-time info.

- LLM Node answers using Gemini's own knowledge.

- Validator Node checks the answer's quality.

- If the answer is valid, it is shown to the user.

- If not, the flow retries via the Supervisor.
