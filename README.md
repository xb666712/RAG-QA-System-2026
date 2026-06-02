# RAG-QA-System

基于Ollama的本地知识库RAG问答系统

## 项目简介

本项目是一个基于Retrieval-Augmented Generation (RAG) 技术的本地知识库问答系统。系统能够读取PDF/DOCX/TXT文档，将其向量化存储到Chroma向量数据库，并利用Ollama大模型进行智能问答。

## 环境要求

- Python 3.10+
- Ollama (已安装qwen2:7b和nomic-embed-text模型)

## 安装步骤

### 1. 安装Ollama

访问 [Ollama官网](https://ollama.com/download) 下载并安装Ollama

```bash
# 下载模型
ollama pull qwen2:7b
ollama pull nomic-embed-text
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行应用

**命令行版本：**
```bash
python rag_cli.py
```

**Web版本：**
```bash
streamlit run app.py
```

## 功能特性

### 文档处理
- 支持PDF、DOCX、TXT格式文档
- 批量读取和文本提取
- 智能文本分块（chunk_size=1000, chunk_overlap=200）

### 向量化存储
- 使用nomic-embed-text嵌入模型
- Chroma向量数据库本地存储
- 持久化存储，跨会话复用

### 智能问答
- 基于向量相似度检索相关文档
- RAG技术结合大模型生成答案
- 多轮对话，支持上下文记忆
- 无相关信息时明确提示

## 技术架构

```
用户问题 → 向量化检索 → 文档片段 → Prompt构建 → Ollama(qwen2:7b) → 答案
              ↓
        Chroma向量库
              ↓
        文档分块存储
```

### 核心组件

- **document_loader.py**: 文档加载和文本提取
- **vector_db.py**: 向量数据库管理
- **rag_chain.py**: RAG问答链实现
- **app.py**: Streamlit Web界面
- **rag_cli.py**: 命令行版本

## 测试问答示例

| 问题 | 预期回答 |
|------|---------|
| 什么是自然语言处理？ | 基于NLP文档内容回答 |
| Transformer架构有哪些组件？ | 基于Transformer文档回答 |
| BERT有哪些预训练任务？ | 基于BERT文档回答 |
| 人工智能的未来趋势是什么？ | 文档中未找到相关答案 |

## 项目结构

```
RAG-QA-System/
├── app.py                    # Streamlit Web应用
├── rag_cli.py                # 命令行版本
├── requirements.txt          # 依赖列表
├── README.md                 # 项目说明
├── utils/
│   ├── __init__.py
│   ├── document_loader.py    # 文档加载器
│   ├── vector_db.py          # 向量数据库
│   └── rag_chain.py          # RAG问答链
└── docs/                     # 文档目录（示例文档）
```

## License

MIT License

## 作者

GitHub: [xb666712](https://github.com/xb666712)
