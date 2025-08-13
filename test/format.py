# Import FAISS class from vectorstore library
from langchain_community.vectorstores import FAISS
# Import OpenAIEmbeddings from the library
from langchain_openai import OpenAIEmbeddings
# Set the OPENAI_API_KEY as the environment variable
import os
os.environ["OPENAI_API_KEY"] = <YOUR_API_KEY>
# Instantiate the embeddings object
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
# Load the database stored in the local directory
vector_store=FAISS.load_local(
folder_path="../../Assets/Data",
index_name="CWC_index",
embeddings=embeddings,
allow_dangerous_deserialization=True
)
# Original Question
query = "Who won the 2023 Cricket World Cup?"
# Ranking the chunks in descending order of similarity
retrieved_docs = vector_store.similarity_search(query, k=2)
# Selecting the first chunk as the retrieved information
retrieved_context= retrieved_docs[0].page_content
# Creating the prompt
augmented_prompt=f"""
Given the context below, answer the question.
Question: {query}
Context : {retrieved_context}
Remember to answer only based on the context provided and not from any other
source.
If the question cannot be answered based on the provided context, say I don't
know.
"""
# Importing the OpenAI library from langchain
from langchain_openai import ChatOpenAI
# Instantiate the OpenAI LLM
llm = ChatOpenAI(
model="gpt-4o-mini",
temperature=0,
max_tokens=None,
timeout=None,
) max_retries=2
# Make the API call passing the augmented prompt to the LLM
response = llm.invoke (
[("human",augmented_prompt)]
)
# Extract the answer from the response object
answer=response.content
print(answer)