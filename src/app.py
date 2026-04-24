import streamlit as st
import pandas as pd

from llm import get_llm
from vectorstore import get_vectorstore
from ingest import load_documents

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="SA Budget RAG", layout="wide")

st.title("🇿🇦 South African Budget RAG Chatbot")
st.subheader("by Lehlohonolo Saohatse")
st.caption("Ask questions or compare budgets (2023–2026)")


# -----------------------------
# SETUP RAG
# -----------------------------
@st.cache_resource
def setup():
    docs = load_documents("data/")
    vectorstore = get_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a South African budget analyst.

Use ONLY the context.

If comparing years, structure output clearly.

Context:
{context}

Question:
{question}
""")

    return retriever, llm, prompt


retriever, llm, prompt = setup()


# -----------------------------
# FORMAT DOCS
# -----------------------------
def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source_file')}] {doc.page_content}"
        for doc in docs
    )


# -----------------------------
# RAG FUNCTION (WITH SOURCES)
# -----------------------------
def ask_rag(question):
    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, docs


# -----------------------------
# SIDEBAR (SOURCES)
# -----------------------------
st.sidebar.title("📄 Retrieved Sources")

if "sources" not in st.session_state:
    st.session_state.sources = []


# -----------------------------
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Ask about the budget or type 'compare budgets'")


if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # -------------------------
    # 🔥 SPECIAL: COMPARISON MODE
    # -------------------------
    if "compare" in user_input.lower():

        with st.spinner("Analyzing budgets..."):

            docs = retriever.invoke(user_input)

            # VERY SIMPLE structured extraction (you can improve later)
            data = []

            for d in docs:
                source = d.metadata.get("source_file")

                # crude extraction example
                text = d.page_content[:500]

                data.append({
                    "Year": source,
                    "Snippet": text
                })

            df = pd.DataFrame(data)

        with st.chat_message("assistant"):
            st.write("### 📊 Budget Comparison Table")
            st.dataframe(df)

    else:
        # -------------------------
        # NORMAL RAG
        # -------------------------
        with st.spinner("Thinking..."):
            answer, docs = ask_rag(user_input)

        # save sources
        st.session_state.sources = docs

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)


# -----------------------------
# SHOW SOURCES IN SIDEBAR
# -----------------------------
if st.session_state.sources:
    for i, doc in enumerate(st.session_state.sources):
        st.sidebar.markdown(f"### Source {i+1}")
        st.sidebar.write(f"📄 {doc.metadata.get('source_file')}")
        st.sidebar.write(doc.page_content[:300] + "...")
