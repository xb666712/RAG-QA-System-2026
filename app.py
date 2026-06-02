import streamlit as st
import os
import tempfile
import logging
import numpy as np

# Fix NumPy 2.0 compatibility issues
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'uint'):
    np.uint = np.uint64

from utils.document_loader import load_documents_from_folder
from utils.vector_db import VectorDBManager
from utils.rag_chain import RAGQAChain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def init_session_state():
    """初始化会话状态"""
    if "vector_db" not in st.session_state:
        try:
            st.session_state.vector_db = VectorDBManager()
            st.session_state.vector_db.init_vector_store()
            logger.info("向量数据库初始化成功")
        except Exception as e:
            st.error(f"向量数据库初始化失败: {str(e)}")
            logger.error(f"向量数据库初始化失败: {str(e)}")
    
    if "rag_chain" not in st.session_state:
        if "vector_db" in st.session_state:
            try:
                st.session_state.rag_chain = RAGQAChain(st.session_state.vector_db)
                logger.info("RAG问答链初始化成功")
            except Exception as e:
                st.error(f"RAG问答链初始化失败: {str(e)}")
                logger.error(f"RAG问答链初始化失败: {str(e)}")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def main():
    """主应用入口"""
    st.set_page_config(
        page_title="RAG问答系统",
        page_icon="📚",
        layout="wide"
    )
    
    # 初始化会话状态
    init_session_state()
    
    # 设置页面样式
    st.markdown(
        """
        <style>
        .main {
            max-width: 1400px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .user-message {
            background-color: #e0f2fe;
            margin-left: 2rem;
        }
        .assistant-message {
            background-color: #f0fdf4;
            margin-right: 2rem;
        }
        .source-tag {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 页面标题
    st.title("📚 RAG问答系统")
    st.markdown("基于Ollama的本地知识库问答系统")
    
    # 创建两列布局
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        # 文档上传区域
        st.subheader("📁 文档管理")
        
        # 文件上传
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="支持PDF、DOCX、TXT格式的文档"
        )
        
        # 构建知识库按钮
        if st.button("🔄 构建知识库", use_container_width=True):
            if uploaded_files:
                with st.spinner("正在处理文档..."):
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # 保存上传的文件到临时目录
                            for f in uploaded_files:
                                file_path = os.path.join(temp_dir, f.name)
                                with open(file_path, "wb") as sf:
                                    sf.write(f.getbuffer())
                            
                            # 加载文档
                            docs = load_documents_from_folder(temp_dir)
                            
                            if docs:
                                # 添加到向量数据库
                                added_count = st.session_state.vector_db.add_documents(docs)
                                st.success(f"✅ 成功添加 {len(docs)} 个文档，共 {added_count} 个文本块")
                                logger.info(f"成功添加 {len(docs)} 个文档，{added_count} 个文本块")
                            else:
                                st.warning("⚠️ 未找到有效文档")
                    except Exception as e:
                        st.error(f"❌ 构建知识库失败: {str(e)}")
                        logger.error(f"构建知识库失败: {str(e)}")
            else:
                st.warning("⚠️ 请先上传文档")
        
        # 知识库统计信息
        st.subheader("📊 知识库统计")
        if "vector_db" in st.session_state and st.session_state.vector_db:
            doc_count = st.session_state.vector_db.get_document_count()
            st.metric("文本块数量", doc_count)
            
            # 已上传的文档列表
            sources = st.session_state.vector_db.get_all_sources()
        if "sources" in locals() and sources:
            st.subheader("📄 已加载文档")
            for source in sources:
                col_info, col_action = st.columns([4, 1])
                col_info.write(source)
                if col_action.button("🗑️", key=f"delete_{source}", help="删除该文档"):
                    try:
                        st.session_state.vector_db.delete_by_source(source)
                        st.success(f"已删除文档: {source}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除文档失败: {str(e)}")
        
        # 清空知识库按钮
        if "doc_count" in locals() and doc_count > 0:
            if st.button("🗑️ 清空知识库", use_container_width=True, type="primary"):
                try:
                    st.session_state.vector_db.clear_database()
                    st.session_state.rag_chain = RAGQAChain(st.session_state.vector_db)
                    st.session_state.chat_history = []
                    st.success("✅ 知识库已清空")
                    logger.info("知识库已清空")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 清空知识库失败: {str(e)}")
                    logger.error(f"清空知识库失败: {str(e)}")
    
    with col2:
        # 问答区域
        st.subheader("💬 智能问答")
        
        # 问题输入
        question = st.text_input(
            "请输入您的问题:",
            placeholder="例如：什么是自然语言处理？",
            help="输入问题后点击提问按钮或按Enter键"
        )
        
        # 提问按钮
        col_ask, col_clear = st.columns([1, 1])
        with col_ask:
            ask_button = st.button("🔍 提问", use_container_width=True, type="primary")
        with col_clear:
            clear_button = st.button("🧹 清空对话", use_container_width=True)
        
        # 处理提问
        if (ask_button or st.session_state.get("submit_question")) and question.strip():
            with st.spinner("正在思考..."):
                try:
                    result = st.session_state.rag_chain.ask(question)
                    st.session_state.chat_history = result["chat_history"]
                    
                    # 显示回答
                    st.success("回答完成！")
                except Exception as e:
                    st.error(f"❌ 回答生成失败: {str(e)}")
                    logger.error(f"回答生成失败: {str(e)}")
        
        # 处理清空对话
        if clear_button:
            st.session_state.rag_chain.clear_history()
            st.session_state.chat_history = []
            st.success("✅ 对话已清空")
        
        # 显示对话历史
        st.subheader("📝 对话历史")
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.markdown(
                        f"""
                        <div class="chat-message user-message">
                            <strong>👤 你：</strong> {content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 助手：</strong> {content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("暂无对话历史，请上传文档并开始提问")

if __name__ == "__main__":
    main()
