import os
import logging

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
# from telegram import Update
# from telegram.ext import (
#     Application,
#     CommandHandler,
#     MessageHandler,
#     CallbackContext,
#     filters,
# )
import asyncio
import logging
import sys
from os import getenv

from aiogram import Bot, Dispatcher, html, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command


load_dotenv()
dp = Dispatcher()

# API_TOKEN = os.getenv('HF_API_TOKEN')
BOT_TOKEN = os.getenv('BOT_TOKEN')
giga_key = os.environ.get("SB_AUTH_DATA")
giga = GigaChat(credentials=giga_key, model="GigaChat", timeout=30,
                verify_ssl_certs=False, max_tokens=500)
# API_URL = (
#     "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1")

user_requests = {}
MAX_REQUETS_PER_DAY = 50
PATH_INDEX = "faiss_index"
# embeddings = HuggingFaceEmbeddings(model_name='all_MiniLM-L6-v2')


def get_rag_database(list_of_urls):
    loader = AsyncHtmlLoader(list_of_urls)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    return docs_transformed


def create_knowledge_base(documents):
    """Creates a vectorstore for the knowledge base."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                   chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="DeepPavlov/rubert-base-cased-sentence")
    # embeddings = GigaChatEmbeddings(verify_ssl_certs=False,
    # scope="GIGACHAT_API_PERS")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


def get_retrieval_chain(vectorstore):
    """Creates a conversational retrieval chain."""

    # 1. Define an LLM
    # llm = ChatOpenAI(model="gpt-4", temperature=0)

    # 2. Define a chain for combining documents
    combine_docs_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. "
            "Use the following context to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )
    combine_docs_chain = load_qa_chain(giga, chain_type="stuff",
                                       prompt=combine_docs_prompt)

    # 3. Define a question generator chain
    question_generator_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "Given the following conversation "
            "history and a follow-up question, rewrite the follow-up "
            "question to make it a standalone question.\n\n"
            "Conversation history:\n{chat_history}\n\n"
            "Follow-up question:\n{question}\n\n"
            "Standalone question:"
        ),
    )
    question_generator_chain = LLMChain(llm=giga,
                                        prompt=question_generator_prompt)

    # 4. Add memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)

    # 5. Build the ConversationalRetrievalChain
    retrieval_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        question_generator=question_generator_chain,
        combine_docs_chain=combine_docs_chain,
        memory=memory,
    )

    return retrieval_chain


# Step 3: Generate Topics and Questions from Vacancy Description
# Define a prompt tem
# plate for extracting interview topics and questions.
TOPICS_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["vacancy"],
    template=(
        "Given the following job description: \n"
        "{vacancy}\n"
        "1. Extract a list of relevant topics that "
        "the candidate should prepare for.\n"
        "2. Provide sample interview questions and "
        "suggested answers for each topic."
    )
)


def generate_topics_and_questions(vacancy_description, retrieval_chain=None):
    """Generates interview topics, questions, and answers
    by combining LLM knowledge and retrieved knowledge base."""
    retrieved_knowledge = ""

    # Step 1: Retrieve knowledge from the knowledge base if
    # the chain is provided
    if retrieval_chain:
        knowledge_results = retrieval_chain({"question": vacancy_description})
        retrieved_knowledge = knowledge_results['answer']

    # Step 2: Combine both sources of knowledge into a single prompt
    # for the LLM
    combined_prompt = (
        "You are an IT interview assistant helping a candidate "
        "prepare for a job interview.\n\n"
        "Job Description:\n"
        f"{vacancy_description}\n\n"
        "Additional Knowledge from the database:\n"
        f"{retrieved_knowledge}\n\n"
        "Based on the job description and the additional knowledge:\n"
        "1. Extract a list of relevant topics the candidate "
        "should prepare for.\n"
        "2. Provide sample interview questions for each topic.\n"
        "3. Provide concise suggested answers for each question."
    )

    # Step 3: Generate the response using the LLM
    response = giga.invoke(combined_prompt).content
    return response


# # Загрузка векторного хранилища
# def load_vector_store(embeddings, vector_store_path="faiss_index"):
#     if not os.path.exists(vector_store_path):
#         raise FileNotFoundError(f"Индекс {vector_store_path} не найден!")
#     try:
#         vector_store = FAISS.load_local(
#             folder_path=vector_store_path,
#             embeddings=embeddings,
#             allow_dangerous_deserialization=True
#         )
#         return vector_store
#     except Exception as e:
#         raise ValueError(f"Ошибка загрузки векторного хранилища: {e}")


# # Инициализация цепочки
# def initialize_chain(vector_store):
#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# Количество возвращаемых документов
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=HuggingFaceAPIWrapper(api_url=API_URL,
#         api_token=API_TOKEN),  # Подключение LLM
#         retriever=retriever
#     )
#     return chain

# Создание промпта
# def create_prompt(user_query, relevant_docs):
#     docs_summary = "\n".join([doc.page_content for doc in
#                               relevant_docs])  # Объединение
#     # текста из документов
#
#     prompt = f"""Ты - ассистент для подготовки к IT собеседованиям.
#     Ответь на вопрос пользователя кратко и точно.
#     Вопрос: {user_query}
#     Релевантная информация: {docs_summary}"""
#
#     return prompt


def process_url(url):
    """Fetches and preprocesses content from a URL."""
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents[0].page_content if documents else ""


# Получение ответа от языковой модели
# def get_assistent_response(user_query, vectore_store, api_url, api_token):
#     retriever = vectore_store.as_retriever(search_kwargs={"k": 1})
#     relevant_docs = retriever.get_relevant_documents(user_query)
#     prompt = create_prompt(user_query, relevant_docs)
#
#     headers = {'Authorization': f'Bearer {api_token}'}
#     payload = {
#         'inputs': prompt,
#         'parameters': {
#             'max_length': 2000,
#             'temperature': 0.6,
#             'num_return_sequences': 1,
#         }
#     }
#
#     response = requests.post(api_url, headers=headers, json=payload)
#     if response.status_code == 200:
#         output = response.json()
#         return output[0]['generated_text'][len(prompt) + 1:]
#     # return output[0]['generated_text'][len(prompt):].strip()
#     else:
#         raise ValueError(
#             f'Ошибка API: {response.status_code}, {response.text}')


# Основная логика Telegram-бота
@dp.message(Command("start"))
async def command_start_handler(message: types.Message) -> None:
    # await update.message.reply_text(
    #     "Привет! Я ваш ассистент по подгтовке к IT собеседованиям. "
    #     "Напишите описание вакансии, и я вам помогу!"
    # )
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}! "
                         f"Я ваш ассистент по подгтовке к IT собеседованиям. "
        "Напишите описание вакансии, и я вам помогу!")


@dp.message(Command("new"))
async def command_new_handler(message: types.Message) -> None:
    await message.answer("Напишите описание вакансии, и я вам помогу!")

@dp.message()
async def handle_message(message: types.Message):
    user_query = message.text

    # try:
    #     vector_store = load_vector_store(embeddings, path_index)
    # except Exception as e:
    #     await update.message.reply_text('Произошла ошибка
    #     при загрузке данных')
    #
    #     return

    try:
        response = generate_topics_and_questions(user_query, retrieval_chain)
        print(f"===\n{response=}\n===")
        # response = get_assistent_response(
        #     user_query, vector_store, api_url=API_URL, api_token=API_TOKEN
        # )
        await message.answer(response)
    except Exception as e:
        await message.answer(f'Произошла ошибка {e}')


async def main() -> None:
    bot = Bot(token=BOT_TOKEN,
              default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    urls = ["https://uproger.com/100-voprosov-c-sobesov-v-data-science-i-ml",
            "https://kalashnikof.com/blog/voprosy-dlya-"
            "sobesedovaniya-frontend-react/"]
    rag_database = get_rag_database(urls)
    vectorstore = create_knowledge_base(rag_database)
    retrieval_chain = get_retrieval_chain(vectorstore)

    asyncio.run(main())
