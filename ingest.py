from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

DATA_PATH = "./data/"
DB_PATH = "./vectorstores/db_faiss"


def create_vector_db():
    loader = DirectoryLoader(path="data/pdfs/",
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device" : "cpu"}
    )
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_PATH)


if __name__=="__main__":
    create_vector_db()