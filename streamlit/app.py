import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import *
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDER,
        model_kwargs={"device" : DEVICE}
    )
    db = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings
    )

    # document_loader = DirectoryLoader(
    #     path=DATA_DIR_PATH,
    #     glob="*.pdf",
    #     loader_cls=PyPDFLoader
    # )
    # docs = document_loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = CHUNK_SIZE,
    #     chunk_overlap = CHUNK_OVERLAP
    # )

    # text_chunks = text_splitter.split_documents(documents=docs)

    faiss_retriever = db.as_retriever(searhc_kwargs=SEARCH_KWARGS)
    # bm25_retriever = BM25Retriever.from_documents(documents=text_chunks)

    # ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5])
    # print("Loaded retriever...")
    # return ensemble_retriever
    return faiss_retriever


def load_chain():

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=INP_VARS
    )

    retriever = load_retriever()

    model_kwargs = {
        "model" : MISTRAL_MODEL_CKPT,
        "model_type" : MODEL_TYPE,
        "temperature" : TEMPERATURE,
        "do_sample" : True,
        "max_new_tokens" : MAX_NEW_TOKENS,
        "top_p" : TOP_P,
        "top_k" : TOP_K,
        "repetition_penalty" : REPETITION_PENALTY
    }

    llm = CTransformers(
        **model_kwargs
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt" : prompt}
    )

    return qa_chain

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
            st.session_state.chain = load_chain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_question = st.chat_input(placeholder="Ask away...")
    if user_question:
        handle_user_query(user_question)

if __name__ == "__main__":
    main()