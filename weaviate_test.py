import weaviate
from dotenv import load_dotenv, find_dotenv
import os
from langchain.vectorstores import Weaviate
from config import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv(find_dotenv())


def get_retriever():
    
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
    weaviate_client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=auth_config
    )

    dir_loader = DirectoryLoader(
        DATA_DIR_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = dir_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    input_text = text_splitter.split_documents(documents=docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDER,
        model_kwargs={"device" : DEVICE}
    )
    weaviate_vectorstore = Weaviate.from_documents(documents=input_text,
                                                   embedding=embeddings,
                                                   client=weaviate_client,
                                                   by_text=False)
    weaviate_retriever = weaviate_vectorstore.as_retriever(search_kwargs=SEARCH_KWARGS)
    bm25_retriever = BM25Retriever.from_documents(input_text)
    ensemble_retriever = EnsembleRetriever(retrievers=[weaviate_retriever, bm25_retriever],
                                           weights=[0.5, 0.5])
    return ensemble_retriever
    
def load_qa_chain():

    prompt_template = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=INP_VARS
    )

    llm = CTransformers(
        model=MISTRAL_MODEL_CKPT,
        model_type=MODEL_TYPE,
        max_new_tokens = MAX_NEW_TOKENS,
        temperature = TEMPERATURE,
        top_p = TOP_P,
        top_k = TOP_K,
        do_sample=True,
        repetition_penalty=REPETITION_PENALTY
    )

    retriever = get_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt" : prompt_template}
    )

    return chain

def handle_user_query(question):
    with st.spinner("Generating answer..."):
        answer = st.session_state.chain(question)
        st.session_state["messages"].append({
            "role" : "user",
            "content" : question
        })
        st.session_state["messages"].append({
            "role" : "assistant",
            "content" : answer["result"]
        })
        for i in range(len(st.session_state["messages"])):
            with st.chat_message(name=st.session_state["messages"][i]["role"]):
                st.markdown(body=st.session_state.messages[i]["content"])
            


def main():
    st.set_page_config(page_title="PTUBot")
    
    st.header("Welcome to PTUBot!")
    

    if "chain" not in st.session_state:
        with st.spinner(text="Loading models..."):
            st.session_state.chain = load_qa_chain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_question = st.chat_input(placeholder="Ask away...")
    if user_question:
        handle_user_query(user_question)

if __name__ == "__main__":
    main()