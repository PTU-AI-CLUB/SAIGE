from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from config import *
import chainlit as cl

def set_custom_prompt() -> PromptTemplate:

    prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                            input_variables=["context", "question"])
    return prompt

def load_llm():
    llm = CTransformers(
        model=LLAMA_MODEL_CKPT,
        model_type=MODEL_TYPE,
        max_new_tokens = MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        top_p=TOP_P,
        top_k=TOP_K,
        do_sample=True
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type=CHAIN_TYPE,
                                           retriever=db.as_retriever(search_kwargs=SEARCH_KWARGS),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt":prompt})
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER,
                                       model_kwargs={"device":DEVICE})
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name="hkunlp/instructor-xl",
    #     model_kwargs={"device" : "cpu"}
    # )
    db = FAISS.load_local(DB_PATH, embeddings)
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
    msg.content = "Hi, Welcome to the SAIGE. Ask away..."
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