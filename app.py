import chainlit as cl

from src.openai_utils.chatmodel import ChatOpenAI
from src.raqa import RetrievalAugmentedQAPipeline
from src.text_utils import CharacterTextSplitter, FileLoader
from src.vectordatabase import VectorDatabase

vector_database = VectorDatabase()
splitter = CharacterTextSplitter()


async def _populate_vector_database(files):
    try:
        loader = FileLoader(documents=files, encoding="utf-8")
        loader_generator = loader.load()
    except Exception as e:
        print(f"Not able to load the docs: {e}")
        raise e
    try:
        chunks = splitter.split_generator(loader_generator)
        vector_database_result = await vector_database.abuild_from_list(chunks)
    except Exception as e:
        print(f"Not able to build the vector database: {e}")
        raise e
    return vector_database_result


@cl.on_chat_start
async def start():
    files = await cl.AskFileMessage(
        content="Upload your file, it can be a PDF or a TXT file.",
        accept=["text/plain", "application/pdf"],
        max_size_mb=10,
    ).send()

    out = cl.Message(content="Loading file...")
    await out.send()

    vector_database = await _populate_vector_database(files)
    cl.user_session.set("vector_database", vector_database)

    out.content = "File loaded!"
    await out.update()


@cl.on_message
async def main(message: cl.Message):
    chat_openai = ChatOpenAI()
    vector_database = cl.user_session.get("vector_database")

    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_database, llm=chat_openai
    )

    out = cl.Message(content="")
    await out.send()

    stream = await retrieval_augmented_qa_pipeline.run_pipeline(message.content)
    async for chunk in stream:
        if token := chunk.choices[0].delta.content or "":
            await out.stream_token(token)

    await out.update()
