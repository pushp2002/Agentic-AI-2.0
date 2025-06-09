import streamlit as st
from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END
import operator
from typing import List, Sequence, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph

# Load environment
load_dotenv()

# Setup models
llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Load & prepare documents
loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50
)
new_docs = text_splitter.split_documents(docs)
db = Chroma.from_documents(new_docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})


# Agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]


# Output parser
class TopicSelection(BaseModel):
    Topic: str = Field(description="selected topic")
    Reasoning: str = Field(description="reasoning behind topic selection")


parser = PydanticOutputParser(pydantic_object=TopicSelection)


# Supervisor node
def supervisor_node(state: AgentState):
    question = state["messages"][-1]
    prompt = PromptTemplate(
        template="""
        Classify the user query into one of these: [Mental Health, Real Time Info, Not Related].
        Rule for classifying:
        If question talks about mental health or some similar information, then classify it to 'Mental health'.
        If question talks about sonme recent or latest information, then classify it to 'Real Time Info'.
        Otherwise, classify it to 'Not Related'.
        Return reasoning too.
        Query: {question}
        {format_instructions}
        """,
        input_variables=["question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )
    chain = prompt | llm_model | parser
    response = chain.invoke({"question": question})
    return {"messages": [response.Topic]}


# Routing logic
def router(state: AgentState):
    last = state["messages"][-1].lower()
    if any(
        word in last for word in ["real time", "live", "latest", "current"]
    ):
        return "Web"
    elif "mental" in last or "mind" in last:
        return "RAG"
    else:
        return "LLM"


# Helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RAG node
def rag_node(state: AgentState):
    question = state["messages"][0]
    prompt = PromptTemplate(
        template="""Use the context below to answer concisely. If unknown, say so.
        Question: {question}
        Context: {context}
        Answer:""",
        input_variables=["context", "question"],
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    result = chain.invoke(question)
    return {"messages": [result]}


# LLM node
def llm_node(state: AgentState):
    question = state["messages"][0]
    prompt = (
        f"Answer the following question using your own knowledge: {question}"
    )
    response = llm_model.invoke(prompt)
    return {"messages": [response.content]}


# Web search node
def web_node(state: AgentState):
    query = state["messages"][0]
    tool = TavilySearchResults()
    response = tool.invoke({"query": query})
    return {"messages": [response[0].get("content")]}


# Validation
def validator_node(state: AgentState):
    answer = state["messages"][-1]
    if "I don't know" in answer or len(answer.strip()) < 10:
        return {"messages": ["__RETRY__"]}
    return {"messages": [answer]}


# Retry router
def retry_router(state: AgentState):
    if state["messages"][-1] == "__RETRY__":
        return "Supervisor"
    return "Final"


# Final output
def final_node(state: AgentState):
    return {"messages": [f"✅ Final Answer: {state['messages'][-1]}"]}


# Create workflow
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("RAG", rag_node)
workflow.add_node("LLM", llm_node)
workflow.add_node("Web", web_node)
workflow.add_node("Validate", validator_node)
workflow.add_node("Final", final_node)
workflow.set_entry_point("Supervisor")
workflow.add_conditional_edges(
    "Supervisor", router, {"RAG": "RAG", "LLM": "LLM", "Web": "Web"}
)
workflow.add_edge("RAG", "Validate")
workflow.add_edge("LLM", "Validate")
workflow.add_edge("Web", "Validate")
workflow.add_conditional_edges(
    "Validate", retry_router, {"Supervisor": "Supervisor", "Final": "Final"}
)
workflow.add_edge("Final", END)
app_graph = workflow.compile()

# Streamlit UI
st.title("LangGraph-powered QA Assistant")

query = st.text_input("Enter your question:")

if query:
    state = {"messages": [query]}
    result = app_graph.invoke(state)
    st.success(result["messages"][-1])
