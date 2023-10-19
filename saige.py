from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from config import *
import chainlit as cl

class SAIGE:

    def __init__(self) -> None:

        self.prompt_template = PROMPT_TEMPLATE
        self.input_variables = INP_VARS
        self.chain_type = CHAIN_TYPE
        self.search_kwargs = SEARCH_KWARGS
        self.embedder = EMBEDDER
        self.model_ckpt = MODEL_CKPT
        self.model_type = MODEL_TYPE
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.top_k = TOP_K
        self.do_sample = DO_SAMPLE
        self.repetition_penalty = REPETITION_PENALTY

        self._setup_utils()

    def _prompt_util(self):
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=self.input_variables
        )
    
    def _llm_util(self):
        self.llm = CTransformers(
            model=self.model_ckpt,
            model_type=self.model_type,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty
        )
    
    def _qa_chain_util(self):
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.db.as_retriever(search_kwargs=self.search_kwargs),
            chain_type=self.chain_type,
            chain_type_kwargs={"prompt" : self.prompt},
            return_source_documents=True
        )
    
    def _setup_utils(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedder,
            model_kwargs={"device" : "cpu"}
        )
        self.db = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=self.embeddings
        )

        self._prompt_util()
        self._llm_util()
        self._qa_chain_util()

    
    def query(self, query: str) -> str:
        response = self.chain({"query" : query})
        return response["result"]




# def set_custom_prompt() -> PromptTemplate:

#     prompt = PromptTemplate(template=PROMPT_TEMPLATE,
#                             input_variables=["context", "question"])
#     return prompt

# def load_llm():
#     llm = CTransformers(
#         model=LLAMA_MODEL_CKPT,
#         model_type=MODEL_TYPE,
#         max_new_tokens = MAX_NEW_TOKENS,
#         temperature=TEMPERATURE,
#         repetition_penalty=REPETITION_PENALTY,
#         top_p=TOP_P,
#         top_k=TOP_K,
#         do_sample=True
#     )
#     return llm

# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                            chain_type=CHAIN_TYPE,
#                                            retriever=db.as_retriever(search_kwargs=SEARCH_KWARGS),
#                                            return_source_documents=True,
#                                            chain_type_kwargs={"prompt":prompt})
#     return qa_chain


# def qa_bot():
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER,
#                                        model_kwargs={"device":DEVICE})
#     db = FAISS.load_local(DB_PATH, embeddings)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)
#     return qa

# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({"query" : query})
#     return response