from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

#load environment variable
load_dotenv()

#read text file
loader = TextLoader('data/smlouva.txt')
loaded_docs = loader.load()

#split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
splitted_docs = text_splitter.split_documents(loaded_docs)

#create embeddings function
embeddings = OpenAIEmbeddings()

#create a chroma db with embeddings
db = Chroma(embedding_function=embeddings,persist_directory='db', collection_name='zaparkujto')

#save docs to chroma db
db.add_documents(splitted_docs)

