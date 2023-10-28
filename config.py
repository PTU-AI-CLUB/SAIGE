DATA_DIR_PATH = "data/pdfs/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
EMBEDDER = "BAAI/bge-large-en-v1.5"
DEVICE = "cpu"
# PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, please just say that you don't know the answer, don't try to make up
# an answer. The questions will be related to Puducherry Technological University. Keep the answers between 3 to 5 lines.
# Context: {context}
# Question: {question}
# Only return the helpful answer below and nothing else.
# Helpful answer: 
# """
PROMPT_TEMPLATE = """**[INST]** Answer the users question only based on the following context in 2 to 3 sentences.
Question: {question} [/INST]
Context: {context}.
Helpful answer: 
"""
INP_VARS = ['context', 'question']
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k': 2}
# MISTRAL_MODEL_CKPT = "mistral-7b-openorca.Q8_0.gguf"
MODEL_CKPT = "llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_TYPE = "llama"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DB_PATH = "vectorstores/db_faiss"
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY=1.15
DO_SAMPLE=True