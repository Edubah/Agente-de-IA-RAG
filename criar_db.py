from langchain_community.document_loaders import PyPDFDirectoryLoader #É passado uma pasta e ele carrega os arquivos em textos
from langchain.text_splitter import RecursiveCharacterTextSplitter #Vai percorrer o arquivo e dividir em chunks
from langchain_chroma.vectorstores import Chroma #Server de banco de dados vetorizado
from langchain_openai import OpenAIEmbeddings #Modelo de embeddings para vetorizar os chunks
from dotenv import load_dotenv #Carregar as variáveis de ambiente do arquivo .env
load_dotenv() #Carregar as variáveis de ambiente do arquivo .env


PASTA_BASE = "base"
#Criar o banco de dados vetorizado
def criar_db(): #ler a pasta Base para ler o PDF(Base de Conhecimento) / Dividir os documentos de texto (chunks) / Vetorizar os chunks com o processo de embedding
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos(): #Ler todos o documento pdf da pasta BASE
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf") #Carregará os documentos, pdf é o padrão.
    documentos = carregador.load() #Ler os documentos
    return documentos #Retorno dos documentos, é uma

def dividir_chunks(documentos): #Vai receber a lista de documentos retornando os chunks
    #Primeiro é necessário saber quantos caracteres tem cada documento
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000, #Tamanho dos chunks
        chunk_overlap=500, #Sobreposição entre os chunks
        length_function=len, #Função para calcular o tamanho dos chunks
        add_start_index=True #Adicionar índice de início
    )
    chunks = separador_documentos.split_documents(documentos) #Dividir os documentos em chunks
    print(len(chunks))
    return chunks

#Vetorização dos chunks
def vetorizar_chunks(chunks): #Aqui você implementaria a lógica para vetorizar os chunks, por exemplo, usando um modelo de embeddings.
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db") #Criar o banco de dados vetorizado com os chunks e embeddings e por na pasta db
    print("Banco de dados vetorizado criado com sucesso!")

criar_db()