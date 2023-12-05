import datetime

import chainlit as cl
import wandb
from wandb.sdk.data_types.trace_tree import Trace

from src.openai_utils.chatmodel import ChatOpenAI
from src.raqa import RetrievalAugmentedQAPipeline
from src.text_utils import CharacterTextSplitter, FileLoader
from src.vectordatabase import VectorDatabase

vector_database = VectorDatabase()
splitter = CharacterTextSplitter()
wandb.init(project="DocuQuery")


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

    try:
        retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
            vector_db_retriever=vector_database, llm=chat_openai
        )
        out = cl.Message(content="")

        await out.send()
        start_time = datetime.datetime.now().timestamp() * 1000

        stream = await retrieval_augmented_qa_pipeline.run_pipeline(message.content)
        async for chunk in stream:
            if token := chunk.choices[0].delta.content or "":
                await out.stream_token(token)

        end_time = datetime.datetime.now().timestamp() * 1000
        await out.update()

        status = "success"
        status_message = (None,)
        response_text = out.content
        model = chat_openai.model_name

    except Exception as e:
        end_time = datetime.datetime.now().timestamp() * 1000
        status = "error"
        status_message = str(e)
        response_text = ""
        model = ""

    root_span = Trace(
        name="DocuQuery",
        kind="llm",
        status_code=status,
        status_message=status_message,
        start_time_ms=start_time,
        end_time_ms=end_time,
        metadata={
            "model": model,
        },
        inputs={
            "system_prompt": retrieval_augmented_qa_pipeline.formatted_system_prompt,
            "user_prompt": retrieval_augmented_qa_pipeline.formatted_user_prompt,
        },
        outputs={"response": response_text},
    )

    root_span.log(name="docuquery_trace")
