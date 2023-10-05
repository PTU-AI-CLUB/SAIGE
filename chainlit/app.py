from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores_baai/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer. The questions will be related to Puducherry Technological University.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer: 
"""

def set_custom_prompt() -> PromptTemplate:

    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=["context", "question"])
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature=0.7,
        repetition_penalty=1.15,
        top_p=0.95,
        top_k=50,
        do_sample=True
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=db.as_retriever(search_kwargs={"k":2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt":prompt})
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                       model_kwargs={"device":"cpu"})
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name="hkunlp/instructor-xl",
    #     model_kwargs={"device" : "cpu"}
    # )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query" : query})
    return response


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot....")
    await msg.send()
    msg.content = "Hi, Welcome to the PTU Bot. Ask away..."
    await msg.send()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):

    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    # await cl.Message(content=answer).send()