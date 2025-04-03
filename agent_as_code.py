
import os
import re
import time
import mlflow
import numpy as np
from datetime import datetime
from hashlib import sha256

from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LanguageModelLike
from databricks_langchain import DatabricksVectorSearch, ChatDatabricks


from mlflow.models import ModelConfig

config: ModelConfig = ModelConfig(development_config="model_config.yaml")

catalog: str = config.get("catalog")
schema: str = config.get("schema")
vector_search_endpoint: str = config.get("vector_search_endpoint")
index_name: str = config.get("index_name")
model_endpoint: str = config.get("model_endpoint")
text_column: str = config.get("text_column")
columns: list[str] = config.get("columns")


llm: LanguageModelLike = ChatDatabricks(model=model_endpoint, temperature=0.1)


@mlflow.trace
def format_context(docs):
    return "\n".join([f"Passage: {d.page_content}" for d in docs])


@mlflow.trace
def extract_user_query_string(chat_messages_array):
    if not chat_messages_array or not isinstance(chat_messages_array[-1], dict):
        return ""
    return chat_messages_array[-1].get("content", "")


@mlflow.trace
def classify_intent(query):
    query = query.lower()
    if any(word in query for word in ["how to", "install", "fix", "replace"]):
        return "diy"
    elif any(word in query for word in ["return", "refund", "policy"]):
        return "returns"
    return "default"


@mlflow.trace
def trim_chat_history(chat_history, max_turns=10):
    if not chat_history:
        return []
    return chat_history[-max_turns:]



## I think you want to consider refactoring this cache. A global here isnt thread safe and will not be persistent
_summarization_cache = {}


@mlflow.trace
def summarize_old_messages(chat_history, summary_model=None):
    if len(chat_history) <= 10:
        return chat_history

    earlier_messages = chat_history[:-8]
    if summary_model:
        summary_key = sha256(str(earlier_messages).encode()).hexdigest()
        if summary_key in _summarization_cache:
            return [_summarization_cache[summary_key]] + chat_history[-8:]

        summary_prompt = [
            {
                "role": "system",
                "content": "Summarize the following conversation between a sales associate and customer.",
            },
            *earlier_messages,
        ]
        try:
            summary_result = summary_model.invoke({"messages": summary_prompt})
            summary_text = (
                summary_result
                if isinstance(summary_result, str)
                else str(summary_result)
            )
            summary = {
                "role": "system",
                "content": f"Summary of earlier conversation: {summary_text}",
            }
            _summarization_cache[summary_key] = summary
        except Exception as e:
            summary = {"role": "system", "content": f"[Summarization failed: {e}]"}
    else:
        summary = {"role": "system", "content": "[Summary omitted for brevity]"}

    return [summary] + chat_history[-8:]


@mlflow.trace
def route_prompt(intent_type: str):
    agents = config.get("agents")
    match intent_type:
        case "diy":
            return agents.get("diy").get("system_prompt")
        case "returns":
            return agents.get("returns").get("system_prompt")
        case _:
            return agents.get("default").get("system_prompt")


retriever = DatabricksVectorSearch(
    endpoint=vector_search_endpoint,
    index_name=index_name,
    # text_column=text_column,
    columns=columns,
).as_retriever(search_kwargs={"k": 8})


def run(chat_input):
    full_chat_history = chat_input.get("messages", [])
    trimmed_history = trim_chat_history(full_chat_history, max_turns=12)
    summarized_history = summarize_old_messages(
        trimmed_history, summary_model=llm
    )

    question = extract_user_query_string(summarized_history)
    intent_type = classify_intent(question)
    prompt = route_prompt(intent_type)
    docs = retriever.invoke(question)
    context = format_context(docs)

    prompt_template = ChatPromptTemplate.from_messages(
        [{"role": "system", "content": prompt}] + summarized_history
    )

    chain = (
        RunnableMap(
            {
                "context": (lambda x: context),
            }
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain


chain: RunnableSequence = RunnableLambda(run)

mlflow.models.set_model(chain)
