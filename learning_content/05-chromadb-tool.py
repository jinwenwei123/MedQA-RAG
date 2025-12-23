import chromadb
from sqlalchemy.orm.collections import collection


# 列出向量库的collection和记录
def list_collection(db_path):
    client = chromadb.PersistentClient(db_path)
    collections = client.list_collections()
    print(f"chromadb: {db_path}有{len(collections)}个")

    for i, col in enumerate(collections):
        print(f"collection {i}: {col.name}，共有{col.count()}")

db_path="./chroma_langchain_db"
list_collection(db_path)

def delete_collection(db_path,collection_name):
    try:
        client = chromadb.PersistentClient(db_path)
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"删除{collection_name}出错，{e}")
