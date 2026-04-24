from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents(folder_path: str):
    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Add metadata so model knows the year
            for doc in docs:
                doc.metadata["source_file"] = file

            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(all_docs)