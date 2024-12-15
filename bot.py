import os

import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackContext,
		Updater,
    filters,
)
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

API_TOKEN = os.getenv('HF_API_TOKEN')
BOT_TOKEN = os.getenv('BOT_TOKEN')
API_URL = ("https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1")

user_requests = {}
MAX_REQUETS_PER_DAY = 50
PATH_INDEX = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name='all_MiniLM-L6-v2')

# Загрузка векторного хранилища
def load_vector_store(embeddings, vector_store_path="faiss_index"):
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Индекс {vector_store_path} не найден!")
    try:
        vector_store = FAISS.load_local(
            folder_path=vector_store_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        raise ValueError(f"Ошибка загрузки векторного хранилища: {e}")

# # Инициализация цепочки
# def initialize_chain(vector_store):
#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Количество возвращаемых документов
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=HuggingFaceAPIWrapper(api_url=API_URL, api_token=API_TOKEN),  # Подключение LLM
#         retriever=retriever
#     )
#     return chain

# Создание промпта
def create_prompt(user_query, relevant_docs):
		docs_summary = "\n".join([doc.page_content for doc in relevant_docs]) # Объединение текста из документов 

		prompt = f"""
		Ты - ассистент для подготовки к IT собеседованиям.  
		Ответь на вопрос пользователя кратко и точно. 
    Вопрос: {user_query}
    Релевантная информация: {docs_summary}
		"""

		return prompt 

# Получение ответа от языковой модели 
def get_assistent_response(user_query, vectore_store, api_url, api_token):
		retriever = vectore_store.as_retriever(search_kwargs={"k": 1})
		relevant_docs = retriever.get_relevant_documents(user_query)
		prompt = create_prompt(user_query, relevant_docs)

		headers = {'Authorization': f'Bearer {api_token}'}
		payload = {
				'inputs': prompt,
				'parameters': {
						'max_length': 2000,
						'temperature': 0.6,
						'num_return_sequences': 1,
				}
		}

		response = requests.post(api_url, headers=headers, json=payload)
		if response.status_code == 200:
				output = response.json()
				return output[0]['generated_text'][len(prompt) + 1:]
				#return output[0]['generated_text'][len(prompt):].strip()
		else: 
				raise ValueError(f'Ошибка API: {response.status_code}, {response.text}')


# Основная логика Telegram-бота
async def start(update: Update, context: CallbackContext):
		await update.message.reply_text(
				"Привет! Я ваш ассистент по подгтовке к IT собеседованиям. Задайте вопрос, и я вам помогу!"
		)

async def handle_message(update: Update, context: CallbackContext, path_index: str = PATH_INDEX):
		user_query = update.message.text

		try:
				vector_store = load_vector_store(embeddings, path_index)
		except Exception as e:
				await update.message.reply_text('Произошла ошибка при загрузке данных')
				
				return 
		
		try: 
				response = get_assistent_response(
						user_query, vector_store, api_url=API_URL, api_token=API_TOKEN
				)
				await update.message.reply_text(response)
		except Exception as e:
				await update.message.reply_text(f'Произошла ошибка {e}')

def main():
		app = Application.builder().token(BOT_TOKEN).build()

		# Добавление обработчиков 
		app.add_handler(CommandHandler('start', start))
		# Обрабатываем текстовы сообщения, исклчая команды 
		app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

		# Запуск бота
		app.run_polling()

if __name__ == '__main__':
		main()