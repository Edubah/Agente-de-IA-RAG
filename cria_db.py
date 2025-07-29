from langchain_community.document_loaders import PyPDFDirectoryLoader #Passa um diretório e ele carrega os arquivos em texto
from langchain.text_splitter import RecursiveCharacterTextSplitter #Para dividir os textos em pedaços menores
from langchain_chroma.vectorstores import Chroma #Para vetorizar os textos
from langchain_openai import OpenAIEmbeddings #Para gerar embeddings dos textos
from dotenv import load_dotenv

load_dotenv()  # Carrega as variáveis de ambiente do arquivo .env
PASTA_BASE = "base"

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)
        
    #Dividir os documentos em pedaços de textos(Chunks)
    #Vetorizar os chunks com o processo de embenddings
    
def carregar_documentos(): #Vai ler todos os documentos pdfs da pasta Base
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf") #Carrega todos os PDFs da pasta Base
    documentos = carregador.load() #Carrega os documentos
    return documentos
    

def dividir_chunks(documentos): #Aqui você implementaria a lógica para dividir os documentos em pedaços menores
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Tamanho máximo de cada pedaço
        chunk_overlap=500,  # Sobreposição entre os pedaços
        length_function=len,  # Função para calcular o comprimento do texto
        add_start_index=True  # Adiciona o índice de início ao pedaço
    )
    chunks = separador_documentos.split_documents(documentos)  # Divide os documentosclear em pedaços
    print(f"Total de chunks criados: {len(chunks)}")  # Exibe o número total de pedaços criados
    return chunks

def vetorizar_chunks(chunks): # Aqui você implementaria a lógica para vetorizar os chunks
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")  # Cria o banco de dados Chroma a partir dos chunks
    print("Banco de dados criado com sucesso!")  # Mensagem de sucesso ao criar o banco de dados
    
criar_db()