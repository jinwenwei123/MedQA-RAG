# 构建索引
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 读取PDF，按照页管理，Document，List[Document]
file_path = "2505.20829v2.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()
# print(len(docs))  # 页数
# print(type(docs[0]))  # 类型
# print(docs[0])  # 内容
"""
page_content='...',
metadata={
    'producer': 'pikepdf 8.15.1', 
    'creator': 'arXiv GenPDF (tex2pdf:)', 
    'creationdate': '', 
    'author': 'Peiyuan Zhi; Peiyang Li; Jianqin Yin; Baoxiong Jia; Siyuan Huang', 
    'doi': 'https://doi.org/10.48550/arXiv.2505.20829', 
    'license': 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/', 
    'ptex.fullbanner': 'This is pdfTeX, Version 3.141592653-2.6-1.40.28 (TeX Live 2025) kpathsea version 6.4.1', 
    'title': 'Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation', 
    'trapped': '/False', 'arxivid': 'https://arxiv.org/abs/2505.20829v2', 
    'source': '2505.20829v2.pdf', 
    'total_pages': 18, 
    'page': 0, 
    'page_label': '1'
    }
"""

# 2. 分割文本，文本段（chunk），Document，List[Document]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 字符
    chunk_overlap=200,  #重叠部分大小
    add_start_index=True,
)

all_splits=text_splitter.split_documents(docs)  # List(Document)
# print(len(all_splits))
# print(type(all_splits))
# print(all_splits[0])
"""
page_content='...' 
metadata={
    'producer': 'pikepdf 8.15.1', 
    'creator': 'arXiv GenPDF (tex2pdf:)', 
    'creationdate': '', 
    'author': 'Peiyuan Zhi; Peiyang Li; Jianqin Yin; Baoxiong Jia; Siyuan Huang', 
    'doi': 'https://doi.org/10.48550/arXiv.2505.20829', 
    'license': 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/', 
    'ptex.fullbanner': 'This is pdfTeX, Version 3.141592653-2.6-1.40.28 (TeX Live 2025) kpathsea version 6.4.1', 
    'title': 'Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation', 
    'trapped': '/False', 'arxivid': 'https://arxiv.org/abs/2505.20829v2', 
    'source': '2505.20829v2.pdf', 
    'total_pages': 18, 
    'page': 0, 
    'page_label': '1', 
    'start_index': 0
    }
"""

# 3. 向量化：每个文本段 <-> 向量，需要嵌入模型
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# vector_0=embeddings.embed_query(all_splits[0].page_content)
# print(len(vector_0))
# print(type(vector_0))
# print(vector_0)

# 4. 向量库：把多个文本段/向量存到向量库
vector_store=Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

ids=vector_store.add_documents(documents=all_splits)

print(len(ids))
print(ids)