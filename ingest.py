from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredPDFLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "./data/"
DB_MINILM_PATH = "./vectorstores_minilm/db_faiss"
DB_IXL_PATH = "./vectorstores_xl/db_faiss"
DB_NLPER_PATH = "./vectorstores_nlper/db_faiss"


def create_vector_db():
    loader = DirectoryLoader(path="data/pdfs/",
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device" : "cpu"}
    )
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_NLPER_PATH)


if __name__=="__main__":
    create_vector_db()