from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

agent = create_agent(model="deepseek:deepseek-chat")

# 第一轮问答
results = agent.invoke({"messages": [{"role": "user", "content": "来一首宋词"}]})
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()
print(type(messages))  # list()

# 第二轮问答
messages.append({"role": "user", "content": "再来"})
results = agent.invoke({"messages": messages})
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()
