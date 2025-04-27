# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langchain",
  "databricks-langchain",
  "databricks-agents",
  "databricks-sdk",
  "mlflow",
  "pydantic",
  "python-dotenv",
  "uv",
  "rich",
  "loguru",
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from importlib.metadata import version

from typing import Sequence

from importlib.metadata import version

pip_requirements: Sequence[str] = [
    f"langchain=={version('langchain')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"mlflow=={version('mlflow')}",
    f"databricks-agents=={version('databricks-agents')}",
    f"pydantic=={version('pydantic')}",
    f"loguru=={version('loguru')}",
]
print("\n".join(pip_requirements))


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC
# MAGIC from typing import Sequence, Any
# MAGIC import sys
# MAGIC import mlflow
# MAGIC from mlflow.models import ModelConfig
# MAGIC
# MAGIC from langchain_core.runnables import RunnableLambda, RunnableSequence
# MAGIC from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# MAGIC from langchain_core.prompt_values import PromptValue
# MAGIC
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC
# MAGIC from loguru import logger
# MAGIC
# MAGIC logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
# MAGIC
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC
# MAGIC llm: LanguageModelLike = ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct")
# MAGIC
# MAGIC def extract_input_values(input_dict: dict[str, Any]) -> dict[str, Any]:
# MAGIC     logger.info("extract_input_values: {input_dict}", input_dict=input_dict)
# MAGIC     user_message = ""
# MAGIC     if "messages" in input_dict and len(input_dict["messages"]) > 0:
# MAGIC         for msg in input_dict["messages"]:
# MAGIC             if msg["role"] == "user":
# MAGIC                 user_message = msg["content"]
# MAGIC                 break
# MAGIC     
# MAGIC     custom_inputs = input_dict.get("custom_inputs", {})
# MAGIC     user_id = custom_inputs.get("user_id", "")
# MAGIC     scd_ids = custom_inputs.get("scd_ids", [])
# MAGIC     store_number = custom_inputs.get("store_num", "") 
# MAGIC     
# MAGIC  
# MAGIC     return {
# MAGIC         "input": user_message,
# MAGIC         "user_id": user_id,
# MAGIC         "store_number": store_number,
# MAGIC         "scd_ids": scd_ids
# MAGIC     }
# MAGIC
# MAGIC prompt_template: ChatPromptTemplate = (
# MAGIC     ChatPromptTemplate.from_template("""
# MAGIC         You are an echo service. Your role is to response to the users question with only the context provided.
# MAGIC         Always include the User ID, Store Number, and SCD Ids in your response.
# MAGIC
# MAGIC         ### Question:
# MAGIC         {input}
# MAGIC
# MAGIC         ### User Id:
# MAGIC         {user_id}
# MAGIC
# MAGIC         ### Store Number:
# MAGIC         {store_number}
# MAGIC
# MAGIC         ### SCD Ids:
# MAGIC         {scd_ids}
# MAGIC
# MAGIC         ### Answer:
# MAGIC
# MAGIC     """)
# MAGIC )
# MAGIC
# MAGIC
# MAGIC chain: RunnableSequence  =  RunnableLambda(extract_input_values) | prompt_template | llm
# MAGIC
# MAGIC
# MAGIC mlflow.models.set_model(chain)
# MAGIC

# COMMAND ----------

example_input = {
    "messages": [
        {
            "role": "user",
            "content": "What is the inventory of the product with id 1?",
        }
    ],
    "custom_inputs": {
        "user_id": "999434",
        "scd_ids": ["7", "4"],
        "store_num": 97,
    },
    "temperature": 0.0,
    "max_tokens": 100,
    "stream": False,
}


# COMMAND ----------

from agent_as_code import chain

messages = chain.invoke(example_input)
messages.content

# COMMAND ----------


from typing import Sequence, Optional, Dict

from dataclasses import dataclass, field, asdict

from mlflow.models.resources import (
    DatabricksResource,
    DatabricksServingEndpoint,
)
import mlflow
from mlflow.models.model import ModelInfo

from mlflow.models import infer_signature
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    StringResponse,
)


@dataclass
class CustomInputs():
    user_id: str = None
    scd_ids: Optional[list[str]] = field(default_factory=list)
    store_num: int = None


# Additional input fields must be marked as Optional and have a default value
@dataclass
class CustomChatCompletionRequest(ChatCompletionRequest):
    custom_inputs: Optional[CustomInputs] = field(default_factory=CustomInputs)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None

sample_input = CustomChatCompletionRequest(
    messages=[{"role": "user", "content": "What is the inventory of the product with id 1?"}],
    custom_inputs=CustomInputs(
        user_id="1",
        scd_ids=["1", "2"],
        store_num=1
    ),

)

signature = infer_signature(asdict(sample_input), StringResponse())

resources: Sequence[DatabricksResource] = [
    DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct")
]

with mlflow.start_run(run_name="chain"):
    mlflow.set_tag("type", "chain")
    logged_agent_info: ModelInfo = mlflow.langchain.log_model(
        lc_model="agent_as_code.py",
        code_paths=[],
        model_config={},
        artifact_path="chain",
        pip_requirements=pip_requirements,
        resources=resources,
        input_example=example_input,
    )

# COMMAND ----------


mlflow.models.predict(
    model_uri=logged_agent_info.model_uri,
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion

registered_model_name: str = "nfleming.ace_hardware.custom_inputs"
mlflow.set_registry_uri("databricks-uc")


model_version: ModelVersion = mlflow.register_model(
    name=registered_model_name,
    model_uri=logged_agent_info.model_uri
)


# COMMAND ----------

from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion

client: MlflowClient = MlflowClient()

client.set_registered_model_alias(name=registered_model_name, alias="Champion", version=model_version.version)
champion_model: ModelVersion = client.get_model_version_by_alias(registered_model_name, "Champion")
print(champion_model)

# COMMAND ----------

from databricks import agents
from databricks.sdk.service.serving import (
    ServedModelInputWorkloadSize, 
)

endpoint_name: str = "ace_custom_inputs"


agents.deploy(
  endpoint_name=endpoint_name,
  model_name=registered_model_name, 
  model_version=champion_model.version, 
  scale_to_zero=True,
  environment_vars={},
  workload_size=ServedModelInputWorkloadSize.SMALL,
  tags={}
)

# COMMAND ----------

example_input

# COMMAND ----------

from typing import Any
from mlflow.deployments import get_deploy_client



response = get_deploy_client("databricks").predict(
  endpoint=endpoint_name,
  inputs=example_input,
)

response

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w: WorkspaceClient = WorkspaceClient()

openai_client = w.serving_endpoints.get_open_ai_client()

response = openai_client.chat.completions.create(
    model=endpoint_name,
    messages=example_input,
)

print (response)

# COMMAND ----------

example_input

# COMMAND ----------

from typing import Any
from mlflow.deployments import get_deploy_client

from agent_as_code import config
from rich import print as pprint


endpoint_name: str = config.get("app").get("endpoint_name")
example_input: dict[str, Any] = config.get("app").get("example_input")

response = get_deploy_client("databricks").predict(
  endpoint=endpoint_name,
  inputs=example_input,
)

pprint(response["messages"])