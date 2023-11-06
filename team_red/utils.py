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
from .prompts_dq import fact_checking_template, qa_template


def set_qa_prompt() -> PromptTemplate:
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"],
    )
    return prompt


def set_fact_checking_prompt() -> PromptTemplate:
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=fact_checking_template,
        input_variables=["context", "question"],
    )
    return prompt


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


def setup_dbqa() -> BaseRetrievalQA:
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG.data.embedding.model,
        model_kwargs={"device": CONFIG.device},
    )
    vectordb = FAISS.load_local(CONFIG.data.embedding.db_path, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


def setup_dbqa_fact_checking() -> BaseRetrievalQA:
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG.data.embedding.model,
        model_kwargs={"device": CONFIG.device},
    )
    vectordb = FAISS.load_local(CONFIG.data.embedding.db_path, embeddings)
    llm = build_llm()
    fact_checking_prompt = set_fact_checking_prompt()
    dbqa = build_retrieval_qa(llm, fact_checking_prompt, vectordb)

    return dbqa
