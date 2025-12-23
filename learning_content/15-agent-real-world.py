### 更复杂一点的智能体 create_agent
## 大模型：model
## 系统提示词：system_prompt - new
## 工具：tools, 用户消息传递参数
#   =>工具运行时上下文传递参数：context_schema - new
## 记忆管理：checkpointer
## 结构化输出：response_format - new

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime

from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# 系统提示词
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.

用中文回答
"""


# 用户消息传递参数
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# 工具运行时上下文传递参数
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# 结构化输出
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# 记忆管理
checkpointer = InMemorySaver()

# 创建智能体
agent = create_agent(
    model="deepseek:deepseek-chat",
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# 配置 thead_id
config = {"configurable": {"thread_id": "1"}}

# 第一轮问答
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response='看来佛罗里达的天气真是"阳光灿烂"啊！这里的天气就像佛罗里达的橙子一样"橙"意满满，阳光普照得让人想"晒"出好心情！不过要小心，这么"晴"朗的天气可能会让你"晒"到不想出门哦！',
#     weather_conditions="It's always sunny in Florida!")

# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)
# ResponseFormat(
#     punny_response='不客气！很高兴能为你"预报"天气！如果你需要更多天气信息，随时"问"我，我会像天气预报一样"准"时为你服务！祝你有个"晴"朗愉快的一天！',
#     weather_conditions=None)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )
