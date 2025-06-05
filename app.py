import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

@st.cache_data(
    show_spinner="Retrieving file..."
)  # only if file is same, it won't run the function again
def retrieve_file(file, api_key):
    # API 키 설정
    os.environ["OPENAI_API_KEY"] = api_key.strip()
    
    # 디렉토리 경로 설정
    cache_dir = "./.cache"
    docs_dir = f"{cache_dir}/document_files"
    embeddings_dir = f"{cache_dir}/document_embeddings/{file.name}"
    file_dir = f"{docs_dir}/{file.name}"
    
    # 필요한 디렉토리들 생성
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    embedding_dir = LocalFileStore(embeddings_dir)
    
    # API 키를 명시적으로 전달
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key.strip(),
        model="text-embedding-ada-002"  # 모델 명시
    )

    # 파일 내용 읽기 및 저장
    file_content = file.read()
    with open(file_dir, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_dir)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    chunks = loader.load_and_split(text_splitter=splitter)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, embedding_dir
    )
    vectorstore = FAISS.from_documents(chunks, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append(
        {"message": message, "role": role}
    )  # Streamlit reruns the entire script from top to bottom on every interaction! So, we need to use session state to store the messages.

def show_message(message, role, save=True):
    with st.chat_message(
        role
    ):  # if use 'with', we can automatically use the functions inside chat_message
        st.markdown(message)  # print the message
    if save:
        save_message(message, role)

def load_messages():
    for message in st.session_state["messages"]:
        show_message(message["message"], message["role"], save=False)

def combine_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)

class LLMCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):  # when llm starts generating new tokens
        self.message_box = st.empty()  # empty widget to update the message
        self.message = ""

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

st.set_page_config(
    page_title="Document GPT",
    page_icon="📃",
)

st.title("Document GPT")

st.markdown(
    """
    Use this chatbot to ask questions to an AI about your files!
    """
)

with st.sidebar:
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="OpenAI API 키를 입력하세요"
    )
    
    file = st.file_uploader("Upload a file")

    st.markdown("---")
    st.markdown("### 📚 소스 코드")
    st.markdown("[GitHub Repository](https://github.com/hakhyun0615/assignment_6)")
    st.markdown("---")

stuff_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. 
            Answer questions using only the following context. 
            \n
            {chunks}
            \n
            Also, remember the conversation history below.
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_message}"),
    ]
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if api_key:
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(openai_api_key=api_key), return_messages=True, memory_key="history"
        )
    memory = st.session_state["memory"]
    
    def load_memory(_):
        return memory.load_memory_variables({})["history"]
    
    if file:
        retriever = retrieve_file(file, api_key)
        show_message("File loaded successfully!", "ai", save=False)
        load_messages()

        user_message = st.chat_input("Ask a question about your file")
        if user_message:
            show_message(user_message, "human")

            chain = (
                {
                    "history": load_memory,
                    "chunks": retriever | RunnableLambda(combine_chunks),
                    "user_message": RunnablePassthrough(),
                }
                | stuff_prompt
                | ChatOpenAI(
                    openai_api_key=api_key,
                    temperature=0.1,
                    streaming=True,
                    callbacks=[LLMCallbackHandler()])
            )
            with st.chat_message("ai"):
                ai_message = chain.invoke(user_message)

            memory.save_context(
                {"input": user_message},
                {"output": ai_message.content},
            )
    else:
        st.session_state["messages"] = []
else:
    st.warning("🔑 OpenAI API 키를 입력해주세요.")
    st.session_state["messages"] = []
    if "memory" in st.session_state:
        del st.session_state["memory"]