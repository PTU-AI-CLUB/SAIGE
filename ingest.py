from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredPDFLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "./data/"
DB_MINILM_PATH = "./vectorstores/db_faiss"
DB_ILARGE_PATH = "./vectorstores/db_faiss"
DB_IXL_PATH = "./vectorstores/db_faiss"
DB_NLPER_PATH = "./vectorstores/db_faiss"


def create_vector_db():
    loader = DirectoryLoader(path="data/",
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs={"device":"cpu"})
    # db = FAISS.from_documents(texts, embeddings)
    # db.save_local(DB_MINILM_PATH)
    
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name="hkunlp/instructor-large",
    #     model_kwargs={"device" : "cpu"}
    # )
    # db = FAISS.from_documents(texts, embeddings)
    # db.save_local(DB_ILARGE_PATH)

    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name="hkunlp/instructor-xl",
    #     model_kwargs={"device" : "cpu"}
    # )
    # db = FAISS.from_documents(texts, embeddings)
    # db.save_local(DB_IXL_PATH)
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device" : "cpu"}
    )
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_NLPER_PATH)


if __name__=="__main__":
    create_vector_db()