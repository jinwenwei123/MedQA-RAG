import json
import pickle
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings


def build_knowledge_documents(candidates_df, question_id_to_content, answer_id_to_content, split_type="train"):
    """构建知识文档"""
    documents = []

    for _, row in candidates_df.iterrows():
        question_id = row['question_id']
        pos_ans_id = row['pos_ans_id']

        # 获取问题和答案内容
        question = question_id_to_content.get(question_id, "")
        pos_answer = answer_id_to_content.get(pos_ans_id, "")

        if question and pos_answer:
            # 创建文档内容（问题和答案组合）
            doc_content = f"问题：{question}\n\n回答：{pos_answer}"

            # 创建元数据
            metadata = {
                "question_id": int(question_id),
                "answer_id": int(pos_ans_id),
                "split": split_type,
                "source": "CMedQAv2",
                "document_type": "qa_pair"
            }

            # 创建LangChain Document对象
            doc = Document(
                page_content=doc_content,
                metadata=metadata
            )
            documents.append(doc)

    return documents


# 分块处理
def chunk_documents(documents, splitter):
    """将文档分块"""
    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            # 为每个块创建新的Document对象
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })

            chunk_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            all_chunks.append(chunk_doc)

    return all_chunks


# 4. 批量添加文档到向量库
def add_documents_to_vectorstore(chunked_documents, vector_store, batch_size=100):
    """批量添加文档到向量库"""
    total_docs = len(chunked_documents)

    for i in range(0, total_docs, batch_size):
        batch = chunked_documents[i:i + batch_size]

        # 提取文本和元数据
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # 添加到向量库
        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=[f"doc_{i + j}" for j in range(len(batch))]
        )

        print(f"已添加 {min(i + batch_size, total_docs)}/{total_docs} 个文档")

    # 持久化保存
    vector_store.persist()
    print("向量库构建完成并已持久化保存")


# 测试检索功能
def test_retrieval(query, vector_store, k=3):
    """测试向量库检索功能"""
    print(f"\n查询：{query}")

    # 执行相似性搜索
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )

    print(f"检索到 {len(results)} 个相关文档：")
    for i, (doc, score) in enumerate(results):
        print(f"\n--- 结果 {i + 1} (相似度分数: {score:.4f}) ---")
        print(f"内容：{doc.page_content[:200]}...")
        print(f"元数据：{doc.metadata}")

    return results


# 保存数据划分信息
def save_data_splits(train_docs, dev_docs, test_docs, question_id_to_content, answer_id_to_content):
    """保存数据划分信息，用于后续评估"""
    data_splits = {
        "train_qa_pairs": [
            {
                "question_id": doc.metadata["question_id"],
                "answer_id": doc.metadata["answer_id"],
                "question": question_id_to_content[doc.metadata["question_id"]],
                "answer": answer_id_to_content[doc.metadata["answer_id"]]
            }
            for doc in train_docs
        ],
        "dev_qa_pairs": [
            {
                "question_id": doc.metadata["question_id"],
                "answer_id": doc.metadata["answer_id"],
                "question": question_id_to_content[doc.metadata["question_id"]],
                "answer": answer_id_to_content[doc.metadata["answer_id"]]
            }
            for doc in dev_docs
        ],
        "test_qa_pairs": [
            {
                "question_id": doc.metadata["question_id"],
                "answer_id": doc.metadata["answer_id"],
                "question": question_id_to_content[doc.metadata["question_id"]],
                "answer": answer_id_to_content[doc.metadata["answer_id"]]
            }
            for doc in test_docs
        ]
    }

    # 创建目录
    os.makedirs("./data_splits", exist_ok=True)

    # 保存为JSON和pickle格式
    with open("./data_splits/data_splits.json", "w", encoding="utf-8") as f:
        json.dump(data_splits, f, ensure_ascii=False, indent=2)

    with open("./data_splits/data_splits.pkl", "wb") as f:
        pickle.dump(data_splits, f)

    print(f"数据划分已保存：")
    print(f"- 训练集QA对：{len(data_splits['train_qa_pairs'])}")
    print(f"- 开发集QA对：{len(data_splits['dev_qa_pairs'])}")
    print(f"- 测试集QA对：{len(data_splits['test_qa_pairs'])}")


def main():
    """主函数"""
    print("开始构建医疗问答向量库...")

    # 加载和预处理数据
    print("加载和预处理数据...")
    # 读取数据
    questions_df = pd.read_csv(r'F:\pythonProject\MedQA-RAG\dataset\cMedQA2\question.csv')
    answers_df = pd.read_csv(r'F:\pythonProject\MedQA-RAG\dataset\cMedQA2\answer.csv')
    train_candidates = pd.read_csv(r'F:\pythonProject\MedQA-RAG\dataset\cMedQA2\train_candidates.txt')
    dev_candidates = pd.read_csv(r'F:\pythonProject\MedQA-RAG\dataset\cMedQA2\dev_candidates.txt')
    test_candidates = pd.read_csv(r'F:\pythonProject\MedQA-RAG\dataset\cMedQA2\test_candidates.txt')

    # 创建问题ID到内容的映射
    question_id_to_content = dict(zip(questions_df['question_id'], questions_df['content']))

    # 创建答案ID到内容的映射
    answer_id_to_content = dict(zip(answers_df['ans_id'], answers_df['content']))
    answer_id_to_question_id = dict(zip(answers_df['ans_id'], answers_df['question_id']))

    # 构建知识文档
    print("构建知识文档...")
    # 构建所有文档
    train_docs = build_knowledge_documents(train_candidates, question_id_to_content, answer_id_to_content, "train")
    dev_docs = build_knowledge_documents(dev_candidates, question_id_to_content, answer_id_to_content, "dev")
    test_docs = build_knowledge_documents(test_candidates, question_id_to_content, answer_id_to_content, "test")

    # 合并所有文档（用于向量库构建）
    all_docs = train_docs + dev_docs + test_docs
    print(f"总文档数量：{len(all_docs)}")
    print(f"训练集：{len(train_docs)}，开发集：{len(dev_docs)}，测试集：{len(test_docs)}")

    # 文本分块
    print("文本分块处理...")
    # 针对中文医疗文本优化的分块器
    chinese_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 适中的块大小
        chunk_overlap=100,  # 重叠确保上下文连贯
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 中文标点优先
        length_function=len,
        keep_separator=True
    )
    # 对文档进行分块
    chunked_docs = chunk_documents(all_docs, chinese_text_splitter)
    print(f"分块后文档数量：{len(chunked_docs)}")

    # 查看分块示例
    print("分块示例：")
    for i, doc in enumerate(chunked_docs[:2]):
        print(f"\n块 {i + 1}:")
        print(f"内容: {doc.page_content[:150]}...")
        print(f"元数据: {doc.metadata}")

    # 构建向量库
    print("构建向量库...")
    # 优化嵌入模型配置
    embedding = OllamaEmbeddings(
        model="qwen3-embedding:8b",
        base_url="http://localhost:11434",
        request_timeout=120,
        batch_size=32
    )

    # 创建Chroma集合，使用持久化客户端确保数据持久保存
    chroma_client = chromadb.PersistentClient(
        path="./chroma_rag_db",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # 创建向量存储
    vector_store = Chroma(
        client=chroma_client,
        collection_name="medical_rag_collection",
        embedding_function=embedding,
        persist_directory="./chroma_rag_db"
    )

    # 执行向量库构建
    add_documents_to_vectorstore(chunked_docs, vector_store)

    # 验证向量库
    print(f"\n向量库统计信息：")
    print(f"集合中文档数量：{vector_store._collection.count()}")

    print("测试...")
    # 示例测试查询
    test_queries = [
        "头痛怎么办？",
        "糖尿病有哪些症状？",
        "感冒应该吃什么药？"
    ]

    for query in test_queries:
        test_retrieval(query, vector_store)

    # 创建检索器实例供后续使用
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # 默认检索5个相关文档
    )

    print("保存数据划分...")
    save_data_splits(train_docs, dev_docs, test_docs, question_id_to_content, answer_id_to_content)

    print("\n✅ 向量库构建完成！")
    print(f"向量库位置：./chroma_rag_db")
    print(f"文档总数：{vector_store._collection.count()}")
    print(f"数据划分已保存至：./data_splits/")


if __name__ == '__main__':
    main()
