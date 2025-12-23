import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

agent = create_agent(model="deepseek:deepseek-chat")

print(agent)
# langgraph.graph.state.CompiledStateGraph
# Graph: nodes - edges 图

print(agent.nodes)
# {'__start__': <langgraph.pregel._read.PregelNode object at 0x00000198204002D0>, 'model': <langgraph.pregel._read.PregelNode object at 0x0000019820400610>}

results = agent.invoke({"messages": [{"role": "user", "content": "What is the weather like in SF"}]})
messages = results["messages"]
print(f"历史消息：{len(messages)}条")

for message in messages:
    message.pretty_print()
"""
{'messages': 
    [
        HumanMessage(content='What is the weather like in SF', additional_kwargs={}, response_metadata={}, id='8ad4cf00-587a-4d4a-8de3-c0851ec82848'), 
        AIMessage(content='As of my last update, I don\'t have real-time data, but I can give you general information about San Francisco\'s weather patterns and suggest how to get the current forecast.\n\n**Typical San Francisco Weather:**\n*   **Cool & Temperate:** Known for its mild, Mediterranean-like climate. It rarely gets extremely hot or cold.\n*   **Microclimates:** Weather can change dramatically from one neighborhood to another. It might be sunny and warm in the Mission District while it\'s foggy and chilly at Ocean Beach.\n*   **"Fog Season":** Late spring through early fall is famous for its fog, especially in western and coastal areas (Karl the Fog!).\n*   **Layered Clothing is Key:** The saying "The coldest winter I ever spent was a summer in San Francisco" (often misattributed to Mark Twain) exists for a reason. Summer afternoons can be warm inland, but temperatures often drop sharply when the fog rolls in.\n\n**For the Accurate, Current Forecast:**\nTo get real-time conditions and the exact forecast for your specific location in the city, I recommend checking:\n*   **Weather Websites/Apps:** [Weather.com](https://weather.com) or [AccuWeather](https://accuweather.com)\n*   **National Weather Service:** [Bay Area Forecast Office](https://www.weather.gov/mtr/)\n*   **A quick web search** for "San Francisco weather" will show you an immediate snapshot.\n\nWould you like me to help you find something specific based on the general climate, like what to pack for an upcoming trip?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 327, 'prompt_tokens': 11, 'total_tokens': 338, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 11}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '0f2b15de-8d08-4e4a-83e2-e4184c83853f', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b4941-8620-7d23-b2d5-5f1f9e114753-0', usage_metadata={'input_tokens': 11, 'output_tokens': 327, 'total_tokens': 338, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})
    ]
}
"""
