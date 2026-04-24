from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_llm
from src.vectorstore import get_vectorstore
from src.ingest import load_documents

# 1. Load docs
docs = load_documents("data")

# 2. Vector DB
vectorstore = get_vectorstore(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


llm = get_llm()


prompt = ChatPromptTemplate.from_template("""
You are a South African budget analyst assistant.

Use ONLY the context below.

If multiple years are present, compare them clearly.

Context:
{context}

Question:
{question}
""")

# 5. Format retrieved docs into text
def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source_file')}] {doc.page_content}"
        for doc in docs
    )

# 6. RAG Chain (LCEL)
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 7. CLI test
if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        if q.lower() == "exit":
            break

        response = rag_chain.invoke(q)
        print("\nAnswer:", response)