import tiktoken

from langchain.chat_models import ChatOpenAI
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain import PromptTemplate, OpenAI, LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import pandas as pd
import os
from dotenv import load_dotenv
import boto3
import io
import tempfile
import argparse
import time


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
#aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
sts_client = boto3.client('sts')

#variables

# Create an argument parser
parser = argparse.ArgumentParser(description='My Script Description')

# Define command-line arguments
parser.add_argument('--input-folder', required=True, help='Path to the input file')
parser.add_argument('--input-file', required=True, help='Path to the output file')

args = parser.parse_args()

input_folder = args.input_folder
input_file = args.input_file


def read_excel_s3(bucket_name,key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    # Specify the S3 bucket name
    s3_bucket_name = bucket_name
    
    
    s3_key = key
    
    # 'https://mini-excels.s3.eu-west-1.amazonaws.com/mini_excels/mini_excelsoutput_1.xlsx'
    
    # Read the Excel file from S3
    s3_response = s3.get_object(Bucket=s3_bucket_name, Key=s3_key)
    excel_data = s3_response['Body'].read()
    
    # You can now parse 'excel_data' using a library like pandas.
    return pd.read_excel(excel_data)

def output_writer(df,nombre,bucket_name):
    data = df
    file_name = nombre
    s3_bucket_name = bucket_name
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_excel_file:
        data.to_excel(temp_excel_file.name, index=False, engine='xlsxwriter')
    
    # Read the temporary file into a binary stream
    with open(temp_excel_file.name, 'rb') as temp_file:
        excel_bytes = io.BytesIO(temp_file.read())
    
    # Specify the S3 key (file path) for the output file
    s3_key = f'output/{file_name}'  # You can customize the path as needed
    
    # Upload the Excel data to S3
    s3.put_object(Bucket=s3_bucket_name, Key=s3_key, Body=excel_bytes)
    
    print(f'Uploaded {file_name} to S3 at s3://{s3_bucket_name}/{s3_key}')

def summarizer(html_texto):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    
    # Map
    map_template = """Aqui tienes el texto del home page de una tienda de ecommerce
    {docs}
    Haz un extenso resumen que caracterize el nicho de la tienda y los puntos importantes a destacar
    """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    summary = map_chain.run(html_texto)
    return summary
    
def nicho_finder(resumen):
    template = """Pregunta: Cual es el nicho de esta tienda online segun el siguiente resúmen del home page:
    
    {resumen}
    
    La respuesta debe resumir el nicho en menos de 5 palabras. Responde única y exclusivamente con el nicho de la tienda, nada mas.
    """
    
    prompt = PromptTemplate(template=template, input_variables=["resumen"])
    
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)
    
    #llm = OpenAI(openai_api_key="YOUR_API_KEY", openai_organization="YOUR_ORGANIZATION_ID")
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    return llm_chain.run(resumen)
    
def advisor(query,summary):
    
    chat = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Template to use for the system message prompt

    template = f"""Transformate en Roger de RojantMedia, un consultor experto en escalar la facturación de tiendas online.
    
            El cliente quiere facturar lo antes posible 20.000€ más al mes, cual es el camino más rápido? que recomendaciones le das personalizadas para su nicho y lo más especificas posibles, usando la información que te he pasado de su tienda online.

            Escribe UNICAMENTE las sugerencias más importantes y nada más.

            Humaniza tu lenguaje, de manera directa y divertida, pero sobretodo que él sienta que le estas dando claves de mucho valor para que quiera implementarlas ya.

            Usa solo los 3 puntos más importantes y estructuralos en (que está haciendo mal y porque eso le impide facturar más, y cual es la solución y que logrará, además acaba aportando datos que demuestren tangiblemente que tienes razón)

            Supon que el cliente ha probado antes a hacer publicidad y no le ha funcionado, esta quemado con este tema... explicale también porque le habrá ido mal

            Supon que el cliente quiere mejorar el valor percibido de su producto para mi cliente potencial y no se cómo venderlos logica y emocionalmente cómo haría un copywritter.
            ----
            INFORMACION TIENDA ONLINE: 

            {summary}

            ----
            
            """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Responde la siguiente pregunta: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response_0 = chain.run(question=query,summary=summary)

    return response_0

df = read_excel_s3(input_folder,input_file)

# Record the start time
start_time = time.time()
##########
#Para hacerlo a traves de un excel y un dataframe
paginas_html = df['paginas']

resumenes = []
for html in paginas_html:
    resumenes.append(summarizer(html))

query = 'Quiero facturar lo antes posible 20.000€ más al mes, cual es el camino más rápido?'
respuestas = []
nichos = []
for resumen in resumenes:
    nichos.append(nicho_finder(resumen))
    respuestas.append(advisor(query,resumen))

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the result in seconds
print(f"Execution time: {float(elapsed_time)/60} minutes")

# Añadimos resultados al df
df['summary'] = resumenes
df['niche'] = nichos
df['personalized_email'] = respuestas

nom = input_file.split('/')[1]
output_writer(df,nom,input_folder)