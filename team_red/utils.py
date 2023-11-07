"""
===========================================
        Module: Util functions
===========================================
"""
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

from .config import CONFIG
from .llm import build_llm


def build_retrieval_qa(  # type: ignore[no-any-unimported]
    llm: CTransformers,
    prompt: PromptTemplate,
    vectordb: FAISS,
) -> BaseRetrievalQA:
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(
            search_kwargs={"k": CONFIG.data.embedding.vector_count}
        ),
        return_source_documents=CONFIG.features.return_source,
        chain_type_kwargs={"prompt": prompt},
    )
    return dbqa
