# =========================
#  Module: Vector DB Build
# =========================
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from team_red.config import CONFIG


# Build vector database
def run_db_build() -> None:
    pdf_loader = DirectoryLoader(CONFIG.data.path, glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(CONFIG.data.path, glob="*.txt", loader_cls=TextLoader)
    documents = pdf_loader.load() + txt_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.data.chunk_size, chunk_overlap=CONFIG.data.chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG.data.embedding.model,
        model_kwargs={"device": CONFIG.device},
    )

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(CONFIG.data.embedding.db_path)


if __name__ == "__main__":
    run_db_build()
