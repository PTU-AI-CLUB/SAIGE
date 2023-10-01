import gradio as gr
from langchain.llms import CTransformers, HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def get_qa():
    # llm = HuggingFaceHub(
    #     repo_id="mistralai/Mistral-7B-v0.1"
    # )
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7
    )
    custom_prompt = """Use only the following pieces of context to answer the question. If you don't know just say don't know. The questions are related to Puducherry Technological University.
    Context: {context}
    Question: {question}
    Helpful Answer: 
    """
    prompt = PromptTemplate(
        template=custom_prompt,
        input_variables=["context", "question"]
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device" : "cpu"}
    )
    vectorstore = FAISS.load_local(folder_path="vectorstores_nlper/db_faiss",
                                   embeddings=embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k":2}),
        chain_type_kwargs={"prompt" : prompt}
    )
    return qa_chain


async def llm_function(message, chat_history):
    qa_chain = get_qa()
    response = await qa_chain.acall(
        message
    )
    print(response.keys())
    output_texts = response["result"]
    return output_texts

title = "Gradio Test"
examples = [
    "Tell me about PTU",
    "What are the departments present in PTU?",
    "Who is the chancellor of PTU?",
    "What are the grades associated with the semester examination?"
]

gr.ChatInterface(
    fn=llm_function,
    title=title,
    examples=examples
).launch()