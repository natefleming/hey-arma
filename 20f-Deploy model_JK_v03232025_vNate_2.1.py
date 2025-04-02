# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = [
  "langchain",
  "databricks-langchain",
  "databricks-vectorsearch",
  "databricks-agents",
  "databricks-sdk",
  "mlflow",
]
pip_requirements = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version


pip_requirements: Sequence[str] = [
  f"langchain=={version('langchain')}",
  f"databricks-langchain=={version('databricks-langchain')}",
  f"databricks-vectorsearch=={version('databricks-vectorsearch')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"mlflow=={version('mlflow')}",
]

print("\n".join(pip_requirements))


# COMMAND ----------

import os 
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host: str = spark.conf.get("spark.databricks.workspaceUrl")
workspace_url: str = f"https://{workspace_host}"
base_url: str = f"{workspace_url}/serving-endpoints/"
token: str = get_databricks_host_creds().token

print(f"workspace_host: {workspace_host}")
print(f"base_url: {base_url}")

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC
# MAGIC import os
# MAGIC import re
# MAGIC import time
# MAGIC import mlflow
# MAGIC import numpy as np
# MAGIC from datetime import datetime
# MAGIC from hashlib import sha256
# MAGIC
# MAGIC from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableSequence
# MAGIC from langchain_core.prompts import ChatPromptTemplate
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from databricks_langchain import DatabricksVectorSearch, ChatDatabricks
# MAGIC
# MAGIC
# MAGIC from mlflow.models import ModelConfig
# MAGIC
# MAGIC config: ModelConfig = ModelConfig(development_config="model_config.yaml")
# MAGIC
# MAGIC catalog: str = config.get("catalog")
# MAGIC schema: str = config.get("schema")
# MAGIC vector_search_endpoint: str = config.get("vector_search_endpoint")
# MAGIC index_name: str = config.get("index_name")
# MAGIC model_endpoint: str = config.get("model_endpoint")
# MAGIC text_column: str = config.get("text_column")
# MAGIC columns: list[str] = config.get("columns")
# MAGIC
# MAGIC
# MAGIC llm: LanguageModelLike = ChatDatabricks(model=model_endpoint, temperature=0.1)
# MAGIC
# MAGIC
# MAGIC def format_context(docs):
# MAGIC     return "\n".join([f"Passage: {d.page_content}" for d in docs])
# MAGIC
# MAGIC
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     if not chat_messages_array or not isinstance(chat_messages_array[-1], dict):
# MAGIC         return ""
# MAGIC     return chat_messages_array[-1].get("content", "")
# MAGIC
# MAGIC
# MAGIC def classify_intent(query):
# MAGIC     query = query.lower()
# MAGIC     if any(word in query for word in ["how to", "install", "fix", "replace"]):
# MAGIC         return "diy"
# MAGIC     elif any(word in query for word in ["return", "refund", "policy"]):
# MAGIC         return "returns"
# MAGIC     return "default"
# MAGIC
# MAGIC
# MAGIC def trim_chat_history(chat_history, max_turns=10):
# MAGIC     if not chat_history:
# MAGIC         return []
# MAGIC     return chat_history[-max_turns:]
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## I think you want to consider refactoring this cache. A global here isnt thread safe and will not be persistent
# MAGIC _summarization_cache = {}
# MAGIC
# MAGIC
# MAGIC def summarize_old_messages(chat_history, summary_model=None):
# MAGIC     if len(chat_history) <= 10:
# MAGIC         return chat_history
# MAGIC
# MAGIC     earlier_messages = chat_history[:-8]
# MAGIC     if summary_model:
# MAGIC         summary_key = sha256(str(earlier_messages).encode()).hexdigest()
# MAGIC         if summary_key in _summarization_cache:
# MAGIC             return [_summarization_cache[summary_key]] + chat_history[-8:]
# MAGIC
# MAGIC         summary_prompt = [
# MAGIC             {
# MAGIC                 "role": "system",
# MAGIC                 "content": "Summarize the following conversation between a sales associate and customer.",
# MAGIC             },
# MAGIC             *earlier_messages,
# MAGIC         ]
# MAGIC         try:
# MAGIC             summary_result = summary_model.invoke({"messages": summary_prompt})
# MAGIC             summary_text = (
# MAGIC                 summary_result
# MAGIC                 if isinstance(summary_result, str)
# MAGIC                 else str(summary_result)
# MAGIC             )
# MAGIC             summary = {
# MAGIC                 "role": "system",
# MAGIC                 "content": f"Summary of earlier conversation: {summary_text}",
# MAGIC             }
# MAGIC             _summarization_cache[summary_key] = summary
# MAGIC         except Exception as e:
# MAGIC             summary = {"role": "system", "content": f"[Summarization failed: {e}]"}
# MAGIC     else:
# MAGIC         summary = {"role": "system", "content": "[Summary omitted for brevity]"}
# MAGIC
# MAGIC     return [summary] + chat_history[-8:]
# MAGIC
# MAGIC
# MAGIC def route_prompt(intent_type: str):
# MAGIC     agents = config.get("agents")
# MAGIC     match intent_type:
# MAGIC         case "diy":
# MAGIC             return agents.get("diy").get("system_prompt")
# MAGIC         case "returns":
# MAGIC             return agents.get("returns").get("system_prompt")
# MAGIC         case _:
# MAGIC             return agents.get("default").get("system_prompt")
# MAGIC
# MAGIC
# MAGIC retriever = DatabricksVectorSearch(
# MAGIC     endpoint=vector_search_endpoint,
# MAGIC     index_name=index_name,
# MAGIC     # text_column=text_column,
# MAGIC     columns=columns,
# MAGIC ).as_retriever(search_kwargs={"k": 8})
# MAGIC
# MAGIC
# MAGIC def run(chat_input):
# MAGIC     full_chat_history = chat_input.get("messages", [])
# MAGIC     trimmed_history = trim_chat_history(full_chat_history, max_turns=12)
# MAGIC     summarized_history = summarize_old_messages(
# MAGIC         trimmed_history, summary_model=llm
# MAGIC     )
# MAGIC
# MAGIC     question = extract_user_query_string(summarized_history)
# MAGIC     intent_type = classify_intent(question)
# MAGIC     prompt = route_prompt(intent_type)
# MAGIC     docs = retriever.invoke(question)
# MAGIC     context = format_context(docs)
# MAGIC
# MAGIC     prompt_template = ChatPromptTemplate.from_messages(
# MAGIC         [{"role": "system", "content": prompt}] + summarized_history
# MAGIC     )
# MAGIC
# MAGIC     chain = (
# MAGIC         RunnableMap(
# MAGIC             {
# MAGIC                 "context": (lambda x: context),
# MAGIC             }
# MAGIC         )
# MAGIC         | prompt_template
# MAGIC         | llm
# MAGIC         | StrOutputParser()
# MAGIC     )
# MAGIC
# MAGIC     #messages = {"messages": summarized_history}
# MAGIC     return chain.invoke({})
# MAGIC
# MAGIC
# MAGIC chain: RunnableSequence = RunnableLambda(run)
# MAGIC
# MAGIC mlflow.models.set_model(chain)

# COMMAND ----------

from rich import print as pprint
from mlflow.models import ModelConfig

config = ModelConfig(development_config="model_config.yaml")


registered_model_name: str = config.get("registered_model_name")
agent_endpoint: str = config.get("agent_endpoint")
input_example = config.get("input_example")
input_example_with_history = config.get("input_example_with_history")


# COMMAND ----------

from agent_as_code import chain


chain.invoke(input_example_with_history)

# COMMAND ----------

from agent_as_code import chain


chain.invoke(input_example)

# COMMAND ----------


import mlflow
from mlflow.models.model import ModelInfo
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
from mlflow.models.resources import DatabricksResource, DatabricksServingEndpoint, DatabricksVectorSearchIndex

from agent_as_code import model_endpoint, index_name

mlflow.set_registry_uri("databricks-uc")

model_signature = mlflow.models.ModelSignature(
    inputs=ChatCompletionRequest,
    outputs=StringResponse
)
 
resources: Sequence[DatabricksResource] = [
  DatabricksServingEndpoint(endpoint_name=model_endpoint),
  DatabricksVectorSearchIndex(index_name=index_name), 
]
 
input_example = config.get("input_example")

with mlflow.start_run():
    logged_agent_info: ModelInfo = mlflow.langchain.log_model(
        lc_model="agent_as_code.py",
        artifact_path="agent",
        model_config=config.to_dict(),
        pip_requirements=pip_requirements,
        resources=resources,
        input_example=input_example,
        signature=model_signature,
    )



# COMMAND ----------

import mlflow


registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=registered_model_name
)


# COMMAND ----------

import time

import mlflow

from databricks.agents import deploy
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedModelInputWorkloadSize,
    EndpointStateReady,
    EndpointStateConfigUpdate,
)


latest_model_version = registered_model_info.version

deployment = deploy(
    endpoint_name=agent_endpoint,
    model_name=registered_model_name,
    model_version=latest_model_version,
    scale_to_zero=True,
    workload_size=ServedModelInputWorkloadSize.MEDIUM,
    environment_vars={},
)

w = WorkspaceClient()
while (
    w.serving_endpoints.get(deployment.endpoint_name).state.ready == EndpointStateReady.NOT_READY
    or w.serving_endpoints.get(deployment.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS
):
    print(".", end="")
    time.sleep(60)

print("✅ Endpoint ready:", deployment.query_endpoint)

# COMMAND ----------

import requests


agent_url: str = f"{base_url}/{agent_endpoint}/invocations" 

headers: dict[str, str] = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.post(agent_url, headers=headers, json=input_example)
print("Status Code:", response.status_code)

try:
    print("Response:", response.json())
except Exception:
    print("Raw Text:", response.text)

