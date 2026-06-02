import os
from utils.document_loader import load_documents_from_folder
from utils.vector_db import VectorDBManager
from utils.rag_chain import RAGQAChain

def main():
    print("RAG问答系统 - 命令行版本")
    vdb = VectorDBManager()
    docs = load_documents_from_folder("./docs")
    if docs:
        vdb.add_documents(docs)
    chain = RAGQAChain(vdb)
    while True:
        q = input("问题 (quit退出): ")
        if q.lower() in ["quit", "exit"]:
            break
        if q.strip():
            r = chain.ask(q)
            print(f"\n答案: {r['answer']}\n")

if __name__ == "__main__":
    main()
