# 安装 langchain openai 库
%pip install langchain-openai==0.3.7
# 从 vectorstore 库导入 FAISS 类
from langchain_community.vectorstores import FAISS
# 从库中导入 OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# 将 OPENAI_API_KEY 设置为环境变量
import os
os.environ["OPENAI_API_KEY"] = <YOUR_API_KEY>
# 实例化嵌入对象
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
# 从本地目录加载存储的数据库
vector_store=FAISS.load_local(
folder_path="../../Assets/Data",
index_name="CWC_index",
embeddings=embeddings,
allow_dangerous_deserialization=True
)
# 原始问题
query = "Who won the 2023 Cricket World Cup?"
# 按相似度降序排列块
retrieved_docs = vector_store.similarity_search(query, k=2)