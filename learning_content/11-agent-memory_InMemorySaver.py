# 消息列表的内存管理
# 通过config实现多会话管理
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

checkpointer = InMemorySaver()

agent = create_agent(
    model="deepseek:deepseek-chat",
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}

# 第一轮问答
results = agent.invoke({"messages": [{"role": "user", "content": "来一首宋词"}]}, config=config)
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()

# 第二轮问答
results = agent.invoke({"messages": [{"role": "user", "content": "再来"}]}, config=config)
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()
