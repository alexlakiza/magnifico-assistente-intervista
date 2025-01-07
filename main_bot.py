import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher, html, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, \
    ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

load_dotenv()
dp = Dispatcher()

BOT_TOKEN = os.getenv('BOT_TOKEN')
giga_key = os.environ.get("SB_AUTH_DATA")
giga = GigaChat(credentials=giga_key, model="GigaChat", timeout=30,
                verify_ssl_certs=False, max_tokens=900)


giga_chain = ConversationChain(
        llm=giga,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )


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
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


def get_retrieval_chain(vectorstore):
    """Creates a conversational retrieval chain."""

    combine_docs_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Ты полезный ассистент для подготовки к "
            "собеседованию в IT-компанию. "
            "Используя контекст подбери список тем для описания вакансии.\n\n"
            "Контекст:\n{context}\n\n"
            "Вакансия:\n{question}\n\n"
            "Ответ:"
        ),
    )
    combine_docs_chain = load_qa_chain(giga, chain_type="stuff",
                                       prompt=combine_docs_prompt)

    # question_generator_prompt = PromptTemplate(
    #     input_variables=["chat_history", "question"],
    #     template=(
    #         "Учитывая историю чата и описание вакансии в вопросе, "
    #         "перепиши описание вакансии так, чтобы помочь подготовиться "
    #         "к собеседованию на эту вакансию.\n\n"
    #         "История чата:{chat_history}\n\n"
    #         "Описание вакансии:{question}"
    #     ),
    # )
    # question_generator_chain = LLMChain(llm=giga,
    #                                     prompt=question_generator_prompt)

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)

    conv_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        question_generator=giga_chain,
        combine_docs_chain=combine_docs_chain,
        memory=memory,
    )

    return conv_chain


def generate_topics_and_questions(vacancy_description,
                                  retrieval_chain=None):
    """Generates interview topics, questions, and answers
    by combining LLM knowledge and retrieved knowledge base."""
    retrieved_knowledge = ""

    # Step 1: Retrieve knowledge from the knowledge base if
    # the chain is provided
    if retrieval_chain:
        try:
            knowledge_results = retrieval_chain({"question":
                                                     vacancy_description})
            retrieved_knowledge = knowledge_results['answer']
        except ValueError:
            retrieved_knowledge = ""

    # Step 2: Combine both sources of knowledge into a single prompt
    # for the LLM
    combined_prompt = (
        "Ты помогаешь подготовиться IT-специалисту к собеседованию на "
        "работу.\n\n"
        "Описание вакансии:\n"
        f"{vacancy_description}\n\n"
        "Дополнительные знания для подготовки из базы знаний:\n"
        f"{retrieved_knowledge}\n\n"
        "Опираясь на описание вакансии и дополнительные знания:\n"
        "1. Приведи список 3 тем для подготовки.\n"
        "2. Приведи по 3 примера вопроса на каждую тему.\n"
        "3. Приведи примеры правильных ответов на вопросы.\n\n"
        "Для каждого твоего ответа ограничение в 3600 символов. "
        "Не пиши в своем ответе ничего, кроме "
        "наименования вакансии, тем, вопросов и ответов. Не задавай "
        "вопросы, не пиши выводы или итоги в своем сообщении"
    )

    print(f"{combined_prompt=}")

    # Step 3: Generate the response using the LLM
    response = giga_chain.invoke(combined_prompt)
    return response['response']


def process_url(url):
    """Fetches and preprocesses content from a URL."""
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents[0].page_content if documents else ""


# Основная логика Telegram-бота
@dp.message(Command("start"))
async def command_start_handler(message: types.Message) -> None:
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}! "
                         f"Я ваш ассистент по подготовке к IT собеседованиям. "
                         "Напишите описание вакансии, и я вам помогу!")


@dp.message(Command("new"))
async def command_new_handler(message: types.Message) -> None:
    await message.answer("Напишите описание вакансии, и я вам помогу!")


@dp.message(F.text.lower() == "ещё темы и вопросы 🚀")
async def more_topics(message: types.Message):
    try:
        response = giga_chain.invoke("User: "
                                     "используя последнее описание вакансии, "
                                     "напиши список новых "
                                     "тем, вопросов на эти темы и "
                                     "(важно) ответы на "
                                     "вопросы."
                                     "Важно, чтобы вопросы для этой вакансии "
                                     "не повторялись. "
                                     "Не задавай вопросы встречные вопросы. "
                                     "Не предлагай темы и вопросы для "
                                     "новой вакансии. Отвечай только про "
                                     "старую вакансию. Отказ не принимается")
        print(response)

        await message.answer(response['response'],
                             reply_markup=keyboard)
    except Exception as e:
        await message.answer(f'Произошла ошибка {e}')


kb = [
    [
        types.KeyboardButton(text="Ещё темы и вопросы 🚀"),
        types.KeyboardButton(text="Новая вакансия 💼")
    ],
]

keyboard = types.ReplyKeyboardMarkup(
    keyboard=kb,
    resize_keyboard=True,
    input_field_placeholder="Выберите дальнейшие действия"
)


@dp.message(F.text.lower() == "новая вакансия 💼")
async def new_vacancy(message: types.Message):
    await message.answer("Напишите описание вакансии, и я вам помогу!",
                         reply_markup=types.ReplyKeyboardRemove())


@dp.message()
async def handle_message(message: types.Message):
    user_query = f"User: {message.text}"

    try:
        response = generate_topics_and_questions(user_query, retrieval_chain)
        print(f"===\n{response=}\n===")

        await message.answer(response,
                             reply_markup=keyboard)
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
    print('5')
    retrieval_chain = get_retrieval_chain(vectorstore)

    asyncio.run(main())
