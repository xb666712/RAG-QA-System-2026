import os
import shutil
import logging
from typing import List, Tuple, Optional, Dict, Any

# Fix chromadb compatibility issues
try:
    import chromadb
    # Add missing config attribute for compatibility
    if not hasattr(chromadb, 'config'):
        class MockConfig:
            class Settings:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            __all__ = ['Settings']
        
        chromadb.config = MockConfig()
except ImportError as e:
    logging.error(f"Failed to import chromadb: {e}")
    raise

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VectorDBManager:
    """
    向量数据库管理器
    
    负责文档的向量化存储、检索和管理
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        初始化向量数据库管理器
        
        Args:
            persist_directory: 持久化存储目录
            embedding_model: 嵌入模型名称
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化组件
        self.embeddings = self._init_embeddings()
        self.text_splitter = self._init_text_splitter()
        self.vector_store = None
        
        logger.info(f"VectorDBManager 初始化完成: 模型={embedding_model}, 存储目录={persist_directory}")
    
    def _init_embeddings(self) -> OllamaEmbeddings:
        """初始化嵌入模型"""
        try:
            return OllamaEmbeddings(model=self.embedding_model)
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            raise
    
    def _init_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """初始化文本分割器"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def _ensure_directory(self):
        """确保存储目录存在"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except Exception as e:
            logger.error(f"创建存储目录失败 {self.persist_directory}: {str(e)}")
            raise
    
    def init_vector_store(self, reset: bool = False):
        """
        初始化向量存储
        
        Args:
            reset: 是否重置现有数据库
        """
        if reset:
            self.clear_database()
        
        self._ensure_directory()
        
        try:
            # 使用最新版chromadb的简化API
            self.vector_store = Chroma(
                collection_name="rag_documents",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            # 直接获取文档数量，避免递归调用
            doc_count = self.vector_store._collection.count() if self.vector_store else 0
            logger.info(f"向量存储初始化成功，文档数量: {doc_count}")
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> int:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表，每个元素为 (文件名, 文件内容) 元组
            
        Returns:
            添加的文档块数量
        """
        if not documents:
            logger.warning("没有提供要添加的文档")
            return 0
        
        if not self.vector_store:
            self.init_vector_store()
        
        chunks = []
        metadatas = []
        
        for filename, text in documents:
            if not text.strip():
                logger.warning(f"跳过空文档: {filename}")
                continue
            
            try:
                text_chunks = self.text_splitter.split_text(text)
                for idx, chunk in enumerate(text_chunks):
                    chunks.append(chunk)
                    metadatas.append({
                        "source": filename,
                        "chunk_index": idx,
                        "total_chunks": len(text_chunks)
                    })
                logger.debug(f"文档 {filename} 分割为 {len(text_chunks)} 个块")
            except Exception as e:
                logger.error(f"分割文档失败 {filename}: {str(e)}")
                continue
        
        if chunks:
            try:
                self.vector_store.add_texts(chunks, metadatas)
                self.vector_store.persist()
                logger.info(f"成功添加 {len(chunks)} 个文档块")
                return len(chunks)
            except Exception as e:
                logger.error(f"添加文档到向量数据库失败: {str(e)}")
                raise
        
        return 0
    
    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            文档和相似度分数的列表
        """
        if not self.vector_store:
            self.init_vector_store()
        
        if not query.strip():
            logger.warning("查询文本为空")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.debug(f"搜索完成，找到 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """获取文档块总数"""
        if not self.vector_store:
            self.init_vector_store()
        
        try:
            count = self.vector_store._collection.count()
            return count
        except Exception as e:
            logger.error(f"获取文档数量失败: {str(e)}")
            return 0
    
    def get_retriever(self, k: int = 3) -> Any:
        """
        获取检索器
        
        Args:
            k: 返回结果数量
            
        Returns:
            向量存储的检索器对象
        """
        if not self.vector_store:
            self.init_vector_store()
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def clear_database(self):
        """清空向量数据库"""
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                self.vector_store = None
                logger.info(f"向量数据库已清空: {self.persist_directory}")
            except Exception as e:
                logger.error(f"清空向量数据库失败: {str(e)}")
                raise
    
    def delete_by_source(self, source: str) -> bool:
        """
        根据来源删除文档
        
        Args:
            source: 文档来源（文件名）
            
        Returns:
            是否删除成功
        """
        if not self.vector_store:
            self.init_vector_store()
        
        try:
            results = self.vector_store._collection.get(
                where={"source": source}
            )
            if results["ids"]:
                self.vector_store._collection.delete(ids=results["ids"])
                self.vector_store.persist()
                logger.info(f"成功删除来源为 {source} 的文档")
                return True
            else:
                logger.warning(f"未找到来源为 {source} 的文档")
                return False
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    
    def get_all_sources(self) -> List[str]:
        """获取所有文档来源"""
        if not self.vector_store:
            self.init_vector_store()
        
        try:
            results = self.vector_store._collection.get()
            sources = list(set(meta.get("source", "") for meta in results.get("metadatas", [])))
            return [s for s in sources if s]
        except Exception as e:
            logger.error(f"获取文档来源失败: {str(e)}")
            return []
