"""
===========================================
        Module: Util functions
===========================================
"""
import box
import yaml
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from .llm import build_llm
from .prompts_dq import fact_checking_template, qa_template

# Import config vars
with open("config/config.yml", "r", encoding="utf8") as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"],
    )
    return prompt


def set_fact_checking_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=fact_checking_template,
        input_variables=["context", "question"],
    )
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": cfg.VECTOR_COUNT}),
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={"prompt": prompt},
    )
    return dbqa


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.MODEL_EMBEDDED_BIN_PATH,
        model_kwargs={"device": cfg.DEVICE},
    )
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


def setup_dbqa_fact_checking():
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.MODEL_EMBEDDED_BIN_PATH,
        model_kwargs={"device": cfg.DEVICE},
    )
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    fact_checking_prompt = set_fact_checking_prompt()
    dbqa = build_retrieval_qa(llm, fact_checking_prompt, vectordb)

    return dbqa
