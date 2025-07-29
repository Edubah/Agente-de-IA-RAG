É a forma de gerar respostas ou o cliente gerar perguntas e a IA responder através de um retorno(Retrieve) que provém de uma base de conhecimentos.

LangChain - ótimo para utilizar o RAG para integração de IA.

Algumas etapas são fundamentais para a criação da IA.

1 - Cliente:
	O qual faz a(s) pergunta(s) para a IA e recebe a resposta.
2 - LLM:
	A IA por trás do processo, onde fará o tratamento das informações, nesse exemplo será usado da OpenAI. 
3 - Banco de Dados Vetorizado:
	Uma base de dados que utilizarei para dividir os dados em "chunks"(pedaços) isso para facilitar a coleta de informações, a velocidade e a qualidade dos dados.
	É usado um processo de 'embenddings' que nada mais é que coletar os dados do texto(chunks - pedaços) e vai vetorizar, converter num vetor de números.
	E isso é possível fazer tanto com o dado do usuário quanto com o conhecimento da base de dados. Depois ele vai comparar com a lista de números das perguntas do usuário, com a lista de números que estão na base de conhecimentos.