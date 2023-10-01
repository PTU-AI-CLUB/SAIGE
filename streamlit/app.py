import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from html_template import css, bot_template, user_template


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device" : "cpu"}
    )
    db = FAISS.load_local(
        folder_path="vectorstores_nlper/db_faiss",
        embeddings=embeddings
    )
    return db

def load_chain():

    custom_prompt = """Use the following pieces of context to answer the question. The questions are related to Puducherry Technological University. If you don't know, say don't know
    Context: {context}
    Question: {question}
    Return helpful answer.
    Answer: 
    """
    prompt = PromptTemplate(
        template=custom_prompt,
        input_variables=["context", "question"]
    )

    db = load_vectorstore()

    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        temperature=0.7,
        max_new_tokens=512,
        repetition_penalty=1.13,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k" : 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt" : prompt}
    )

    return qa_chain

def handle_user_query(question):
    st.session_state.chat_history.append(question)
    st.spinner("Generating answer...")
    answer = st.session_state.chain(question)
    st.session_state.chat_history.append(answer["result"])
    for i in range(len(st.session_state.chat_history)):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", st.session_state.chat_history[i]),
                     unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", st.session_state.chat_history[i]),
                     unsafe_allow_html=True)
            


def main():
    st.set_page_config(page_title="PTUBot")
    st.write(css, unsafe_allow_html=True)
    
    st.header("Welcome to PTUBot!")
    

    if "chain" not in st.session_state:
        st.spinner(text="Loading models...")
        st.session_state.chain = load_chain()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask away...")
    if user_question:
        handle_user_query(user_question)

if __name__ == "__main__":
    main()