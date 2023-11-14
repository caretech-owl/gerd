"""
===========================================
        Module: Util functions
===========================================
"""
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

from .config import CONFIG


def build_retrieval_qa(  # type: ignore[no-any-unimported]
    llm: CTransformers,
    prompt: PromptTemplate,
    vectordb: FAISS,
    vector_count: int,
    return_source: bool,
) -> BaseRetrievalQA:
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": vector_count}),
        return_source_documents=return_source,
        chain_type_kwargs={"prompt": prompt},
    )
    return dbqa
