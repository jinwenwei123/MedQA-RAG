import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

model = init_chat_model(
    model="deepseek:deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3
)

for chunk in model.stream("来一段毛主席诗词"):
    print (chunk.content, end='', flush=True)