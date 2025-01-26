
from flask import Flask, render_template, request, jsonify, Response


#from transformers import AutoModelForCausalLM, AutoTokenizer
#import torch

#chatbot librerias
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
#from langchain.document_loaders import PyPDFDirectoryLoader
#from langchain.document_loaders import DirectoryLoader
#from langchain.document_loaders import TextLoader
#from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
#import textwrap
#**************************

#tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


os.environ['OPENAI_API_KEY'] = ''

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

#pdf_path = r'C:\Users\Usuario\Desktop\FlaskProjects\chatbot\pdf\whatsapp'
pdf_path = os.path.join('pdf', 'whatsapp')

database = FAISS.load_local(pdf_path,embeddings)


application = Flask(__name__)

@application.route("/")
def index():
    return render_template('chat.html')


@application.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(database,input,k=3)


def get_Chat_response(db, query, k=3):
    
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about documents based on the documents context: {docs}
        
        Only use the factual information from the transcript to answer the question.

        You must answer in spanish
        
        If you feel like you don't have enough information to answer the question, say "No tengo informacion para responder esta pregunta".
        
        Your answers should be schematic.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response_0 = chain.run(question=query, docs=docs_page_content)
    #response = response.replace("\n", "<br>")
    #response = f'<p>{response}</p>'
    return response_0


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
    #application.run()