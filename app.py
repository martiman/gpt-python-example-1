import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

#load environment variables
openai_api_key = st.secrets["openai_api_key"]

#initialization
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = Chroma(embedding_function=embedding, persist_directory='db', collection_name='zaparkujto')

#application title
st.write('# Ptej se na aplikaci Zaparkujto stejně jako v ChatGPT')

#form input
title = st.text_input('Jaký máš dotaz?', placeholder='např. "K čemu slouží aplikace Zaparkujto?"')

#Processing form input
if title:   
    #find similar paragraphs 
    docs = vectordb.similarity_search(title, k=4)
    
    #ask open ai gpt model
    chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=openai_api_key), chain_type="stuff")
    output = chain({"input_documents": docs, "question": title})
    
    #show output to the application
    st.write(output['output_text'])