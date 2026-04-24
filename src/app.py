import streamlit as st

from llm import get_llm
from vectorstore import get_vectorstore
from ingest import load_documents

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="SA Budget RAG Chatbot",
    page_icon="📊",
    layout="wide"
)

st.title("South African Budget RAG Chatbot")
st.subheader("by Lehlohonolo Saohatse")
st.caption("Ask questions about South Africa’s national budgets (2023–2026)")


# -----------------------------
# Load data (cached so it doesn't reload every refresh)
# -----------------------------
@st.cache_resource
def setup_rag():
    docs = load_documents("data/")  
    vectorstore = get_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a South African budget analyst assistant.

Use ONLY the context below.
If multiple years exist, compare them clearly.

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source_file')}] {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = setup_rag()


# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User input
user_input = st.chat_input("Ask something about the SA budget...")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # get response
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_input)

    # show assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)