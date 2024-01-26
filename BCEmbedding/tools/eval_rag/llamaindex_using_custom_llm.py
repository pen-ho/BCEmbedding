#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : llamaindex_using_custom_llm.py
# Date    : 2023-12-12
from typing import Optional, List, Mapping, Any

from llama_index import ServiceContext, SimpleDirectoryReader, SummaryIndex
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
import openai
from loguru import logger
logger.disable('DEBUG')


class OurLLM(CustomLLM):
    """
    定义自己部署的llm
    """
    context_window: int = 2000
    num_output: int = 256
    model_name: str = "chinese-llama-alpaca-2"
    dummy_response: str = "My response"
    openai.api_key = "EMPTY"
    openai.api_base = "http://192.168.38.28:30392/v1"  # prd

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def get_response(self, prompt):
        """get llm response"""
        completion = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        # logger.info(completion)
        return completion.choices[0].message.content

    # @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_txt = self.get_response(prompt=prompt)
        logger.debug(f'prompt-{len(prompt)}:{prompt}\nresponse-{len(response_txt)}:{response_txt}')
        self.dummy_response = response_txt
        return CompletionResponse(text=self.dummy_response)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
llm = OurLLM()
print(llm)
if __name__ == '__main__':
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model="local:/Users/penho/Documents/nlp/data/pretrained_model/bce-embedding-base_v1"
    )

    # Load the your data
    # documents = SimpleDirectoryReader("./data").load_data()
    documents = SimpleDirectoryReader(input_files=["/Users/penho/Documents/nlp/PythonProject/bm_llm/sample/【优化】1-挂式空调洗前准备岗位观察检查表V1.3-20211029.docx"]).load_data()
    print(f'documents:{documents}')
    index = SummaryIndex.from_documents(documents, service_context=service_context)

    # Query and print response
    query_engine = index.as_query_engine()
    response = query_engine.query("目标是什么")
    print(f'response:{response}')  # 目标是确保空调的清洁和维护，以确保其正常运行并延长其使用寿命。
