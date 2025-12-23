from dotenv import load_dotenv
from langchain.agents import create_agent
from pyexpat.errors import messages

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

messages_list = [{"role": "user", "content": "What is the weather like in SF"}]
# for event in agent.stream(messages_dict, stream_mode="values"):  # 消息
#     messages = event["messages"]
#     print(f"历史消息：{len(messages)}条")
#     # for message in messages:
#     #     message.pretty_print()
#     messages[-1].pretty_print()

for chunk in agent.stream({"messages": messages_list}, stream_mode="messages"):  # token
    print(chunk[0].content, end='')
"""
(
AIMessageChunk(content='I', additional_kwargs={}, response_metadata={'model_provider': 'deepseek'}, id='lc_run--019b4a36-b84a-7103-9c80-daa54cc15b91'), 
{
    'langgraph_step': 1, 
    'langgraph_node': 'model', 
    'langgraph_triggers': ('branch:to:model',), 
    'langgraph_path': ('__pregel_pull', 'model'), 
    'langgraph_checkpoint_ns': 'model:10c41b7f-f181-20d1-7287-98f77d6d3516', 
    'checkpoint_ns': 'model:10c41b7f-f181-20d1-7287-98f77d6d3516', 
    'ls_provider': 'deepseek', 
    'ls_model_name': 'deepseek-chat', 
    'ls_model_type': 'chat', 
    'ls_temperature': None}
)   
"""
# messages = chunk["messages"]
# print(f"历史消息：{len(messages)}条")
# # for message in messages:
# #     message.pretty_print()
# messages[-1].pretty_print()
