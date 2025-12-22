from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import chain

# 嵌入模型
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# 向量库（知识库）
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# 相似度查询
results = vector_store.similarity_search(
    "What imitations does Force-aware have?"
)

for idx, result in enumerate(results):
    print(idx)
    print(result.page_content[:100])
    # print(type(result))  # document

# 带分数的相似度查询
results = vector_store.similarity_search_with_score(
    "What imitations does Force-aware have?"
)

for idx, (doc, score) in enumerate(results):
    print(idx)
    print("score: ", score)  # 分数越低，相似度越高
    print(doc.page_content[:100])
    # print(type(result))  # document

# 用向量进行相似度查询
vector = embeddings.embed_query("What imitations does Force-aware have?")

results = vector_store.similarity_search_by_vector(vector)

for idx, result in enumerate(results):
    print(idx)
    print(result.page_content[:100])


# 封装成chain:langchain:大模型，提示词模板，tools，output，runnable
print("用检索器进行相似度查询：")
@chain
def retriever(query: str) -> list[Document]:
    return vector_store.similarity_search(query, k=1)

results=retriever.invoke("What imitations does Force-aware have?")

for idx, result in enumerate(results):
    print(idx)
    print(result.page_content[:100])