from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

#load environment variables
load_dotenv()

#initialization
question = "Co vše umí aplikace Zaparkujto?"
embedding = OpenAIEmbeddings()
vectordb = Chroma(embedding_function=embedding, persist_directory='db', collection_name='zaparkujto')

#search for similar paragraphs
docs = vectordb.similarity_search(question)

#load question answering chain into OpenAI GPT model
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
output = chain({"input_documents": docs, "question": question})

print(output['output_text'])