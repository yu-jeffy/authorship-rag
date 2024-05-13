# Linux Sqlite >3.35 fix
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()

client = OpenAI()

api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="llama3-70b-8192", api_key=api_key)

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("chapters")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
filename_fn = lambda filename: {"file_name": filename}

# automatically sets the metadata of each document according to filename_fn
documents = SimpleDirectoryReader(
    "./chapters/", file_metadata=filename_fn
).load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=4,
    verbose=True
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(llm=llm, response_mode="refine")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
# response = query_engine.query("What happens in chapter 3?")
# print(response)

SYSTEM_PROMPT = """
You are assisting in authoring a novel. You are tasked with brainstorming and helping to write the book.
"""

# Initialize the conversation with a system message
messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]

while True:
    # Get user input
    user_input = input("User: ")

    # Add user message to the conversation history
    messages.append(ChatMessage(role="user", content=user_input))

    # Convert messages into a string
    message_string = "\n".join([f"{message.role}: {message.content}" for message in messages])

    # Get llm response
    resp = query_engine.query(message_string)

    # Add llm message to the conversation history
    messages.append(ChatMessage(role="assistant", content=resp))

    # Print llm response
    print("llm: ", resp)