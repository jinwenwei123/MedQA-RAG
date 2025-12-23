# checkpointer: 检查点管理器
# checkpoint: 检查点，状态图的总体状态快照
# thread_id: 记忆管理
# 作用: 记忆管理、时间旅行、pause、容错

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add


# 表达状态: 整个状态图的状态
class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


# 构建状态图
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 检查点管理器
checkpointer = InMemorySaver()

# 编译
graph = workflow.compile(checkpointer=checkpointer)

# 配置
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# 运行
results = graph.invoke({"foo": ""}, config)
print(results)
# {'foo': 'b', 'bar': ['a', 'b']}

# 状态查看
print(graph.get_state(config=config))
"""
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']}, next=(), 
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0dff62-a3c0-6b82-8002-6fee389ee851'}}, 
    metadata={'source': 'loop', 'step': 2, 'parents': {}}, 
    created_at='2025-12-23T11:54:41.201241+00:00', 
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0dff62-a3c0-6b81-8001-d2a91c37e8c9'}}, 
    tasks=(), 
    interrupts=()
)
"""

for checkpoint_tuple in checkpointer.list(config):
    print()
    print(checkpoint_tuple)
