import logging
from typing import List, Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from .vector_db import VectorDBManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RAGQAChain:
    """
    RAG问答链
    
    负责将用户问题与知识库进行匹配，并生成回答
    """
    
    def __init__(
        self,
        vdb: VectorDBManager,
        llm_model: str = "llama3:8b",
        max_history: int = 10,
    ):
        """
        初始化RAG问答链
        
        Args:
            vdb: 向量数据库管理器
            llm_model: 大语言模型名称
            max_history: 最大对话历史长度
        """
        self.vdb = vdb
        self.llm_model = llm_model
        self.max_history = max_history
        self.chat_history = []
        self.chain = None
        
        # 初始化LLM
        self.llm = self._init_llm()
        
        # 定义提示词模板
        self.prompt = PromptTemplate.from_template(
            """
            你是一个专业的知识库问答助手。请根据提供的参考文档来回答用户的问题。

            参考文档：
            {context}

            对话历史：
            {chat_history}

            用户问题：
            {question}

            回答要求：
            1. 优先使用参考文档中的信息进行回答
            2. 如果参考文档中没有相关信息，请明确说明"文档中未找到相关答案"
            3. 回答要简洁明了，直接针对问题
            4. 如果有多个相关文档，请综合所有信息进行回答
            5. 不要编造信息

            回答：
            """
        )
        
        logger.info(f"RAGQAChain 初始化完成: 模型={llm_model}")
    
    def _init_llm(self) -> Ollama:
        """初始化大语言模型"""
        try:
            return Ollama(model=self.llm_model)
        except Exception as e:
            logger.error(f"初始化LLM失败: {str(e)}")
            raise
    
    def _format_context(self, documents: List[Document]) -> str:
        """格式化参考文档上下文"""
        if not documents:
            return "无相关参考文档"
        
        formatted_docs = []
        for idx, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            total_chunks = doc.metadata.get("total_chunks", 1)
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            
            formatted_docs.append(
                f"【文档{idx}】来源: {source} (片段 {chunk_idx + 1}/{total_chunks})\n{content}"
            )
        
        return "\n\n".join(formatted_docs)
    
    def _format_chat_history(self) -> str:
        """格式化对话历史"""
        if not self.chat_history:
            return "无对话历史"
        
        history_str = []
        for item in self.chat_history[-self.max_history:]:
            role = "用户" if item["role"] == "user" else "助手"
            history_str.append(f"{role}: {item['content']}")
        
        return "\n".join(history_str)
    
    def _update_chat_history(self, role: str, content: str):
        """更新对话历史"""
        self.chat_history.append({"role": role, "content": content})
        
        # 保持历史长度限制
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2:]
    
    def init_chain(self):
        """初始化问答链"""
        try:
            retriever = self.vdb.get_retriever(k=3)
            
            def context_transformer(inputs: Dict[str, Any]) -> str:
                """转换检索结果为上下文文本"""
                if isinstance(inputs, list):
                    return self._format_context(inputs)
                return ""
            
            # 构建问答链
            self.chain = (
                {
                    "context": retriever | context_transformer,
                    "chat_history": RunnablePassthrough() | (lambda x: self._format_chat_history()),
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("问答链初始化成功")
        except Exception as e:
            logger.error(f"初始化问答链失败: {str(e)}")
            raise
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        回答用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            包含回答、来源和对话历史的字典
        """
        if not question.strip():
            logger.warning("用户问题为空")
            return {
                "answer": "请输入有效的问题",
                "sources": [],
                "chat_history": self.chat_history.copy()
            }
        
        if not self.chain:
            self.init_chain()
        
        try:
            logger.info(f"处理用户问题: {question[:50]}...")
            
            # 搜索相关文档获取来源信息
            search_results = self.vdb.search(question, k=3)
            sources = [
                {
                    "source": doc.metadata.get("source", "未知来源"),
                    "chunk": doc.metadata.get("chunk_index", 0),
                    "similarity": score
                }
                for doc, score in search_results
            ]
            
            # 调用问答链生成回答
            answer = self.chain.invoke(question)
            
            # 更新对话历史
            self._update_chat_history("user", question)
            self._update_chat_history("assistant", answer)
            
            logger.info(f"回答生成完成，长度: {len(answer)}")
            
            return {
                "answer": answer,
                "sources": sources,
                "chat_history": self.chat_history.copy()
            }
        except Exception as e:
            logger.error(f"回答生成失败: {str(e)}")
            return {
                "answer": f"回答生成失败: {str(e)}",
                "sources": [],
                "chat_history": self.chat_history.copy()
            }
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.chat_history.copy()
    
    def set_max_history(self, max_history: int):
        """设置最大对话历史长度"""
        self.max_history = max_history
        # 截断超出部分
        if len(self.chat_history) > max_history * 2:
            self.chat_history = self.chat_history[-max_history * 2:]
        logger.info(f"最大对话历史长度已设置为: {max_history}")
