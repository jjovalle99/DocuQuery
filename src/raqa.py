from src.openai_utils.chatmodel import ChatOpenAI
from src.openai_utils.prompts import SystemRolePrompt, UserRolePrompt
from src.vectordatabase import VectorDatabase

RAQA_PROMPT_TEMPLATE = """
Use the provided context to answer the user's query.

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".

Context:
{context}
"""

raqa_prompt = SystemRolePrompt(RAQA_PROMPT_TEMPLATE)

USER_PROMPT_TEMPLATE = """
User Query:
{user_query}
"""

user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)


class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.formatted_system_prompt = None
        self.formatted_user_prompt = None

    def run_pipeline(self, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        self.formatted_system_prompt = raqa_prompt.create_message(
            context=context_prompt
        )
        self.formatted_user_prompt = user_prompt.create_message(user_query=user_query)

        return self.llm.run([self.formatted_system_prompt, self.formatted_user_prompt])
