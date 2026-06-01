import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# ChatHuggingFace will inject the tokenizer templates locally
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)