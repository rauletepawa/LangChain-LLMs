from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

os.environ['OPENAI_API_KEY'] = ''

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def create_db_from_pdf(pdf_path):
    loader = DirectoryLoader(pdf_path,glob="**/*.pdf", loader_cls = PyPDFLoader)
    #loader = PyPDFDirectoryLoader("pdf_path")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    db = FAISS.from_documents(pages, embeddings)

    return db



def get_response_from_query(db, query, k=3):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

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

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "<br>")
    response = f'<p>{response}</p>'
    return response, docs

pdf_path = r'C:\Users\Usuario\Desktop\FlaskProjects\chatbot\pdf\whatsapp'

#faiss_db = create_db_from_pdf(pdf_path)
#faiss_db.save_local(pdf_path)

#docs = create_db_from_pdf(pdf_path)
#Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=pdf_path)
#chroma_db = Chroma(persist_directory=r'C:\Users\Usuario\Desktop\FlaskProjects\chatbot\pdf\whatsapp',embedding_function=embeddings)

database = FAISS.load_local(pdf_path,embeddings)

while True:
    query = input('Escriba su pregunta: ')
    if query != 'q':
        respuesta = get_response_from_query(database, query, k=3)[0]
        print('\n')
        print(respuesta)
    else:
        break