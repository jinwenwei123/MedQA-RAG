from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()


def get_weather(city: str) -> str:
    """
    Get weather for a given city
    :param city: a given city
    :return: weather for the given city
    """
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[get_weather],
)

print(agent)
print(agent.nodes)
"""
{'__start__': <langgraph.pregel._read.PregelNode object at 0x000001D7767A5AD0>, 
'model': <langgraph.pregel._read.PregelNode object at 0x000001D7767BCED0>, 
'tools': <langgraph.pregel._read.PregelNode object at 0x000001D7767E2450>}
"""

results = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather like in SF"}]})  # How many people in SF?
messages = results["messages"]
print(f"历史消息：{len(messages)}条")

for message in messages:
    message.pretty_print()
