import os
import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain

# 채팅 기록 저장 함수
def save_chat_history(title=""):
    if 'messages' in st.session_state and len(st.session_state.messages) > 0:
        if not os.path.exists('chat_history'):
            os.makedirs('chat_history')
        
        if not title:
            title = "chat_history"
        
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        filename = f"{safe_title}.txt"
        filepath = os.path.join('chat_history', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for msg in st.session_state.messages:
                f.write(f"[{msg['role'].upper()}] {msg['content']}\n")
        
        st.success(f"✅ 채팅 기록이 저장되었습니다: {filename}")
        return True
    else:
        st.warning("저장할 채팅 기록이 없습니다.")
        return False

# 저장된 채팅 기록 표시 함수
def display_saved_chats():
    st.subheader("📁 저장된 채팅 기록")
    
    if not os.path.exists('chat_history'):
        st.info("채팅 기록 폴더가 존재하지 않습니다.")
        return []
        
    files = [f for f in os.listdir('chat_history') if f.endswith('.txt')]
    
    if not files:
        st.info("아직 저장된 채팅 기록이 없습니다.")
        return []
        
    cols = st.columns(3)
    for idx, file in enumerate(files):
        with cols[idx%3]:
            with open(os.path.join('chat_history', file), 'r', encoding='utf-8') as f:
                content = f.read()
            st.download_button(
                label=f"📄 {file}",
                data=content,
                file_name=file,
                mime="text/plain"
            )
    
    return files

# 채팅 기록 불러오기 함수
def load_chat_history(filename):
    messages = []
    file_path = os.path.join('chat_history', filename)
    
    if not os.path.exists(file_path):
        st.error(f"{filename} 파일을 찾을 수 없습니다.")
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("[USER]"):
                messages.append({"role": "user", "content": line[len("[USER] "):].strip()})
            elif line.startswith("[ASSISTANT]"):
                messages.append({"role": "assistant", "content": line[len("[ASSISTANT] "):].strip()})
                
    if messages:
        st.session_state['messages'] = messages
        st.success(f"✅ {filename} 채팅 기록을 불러왔습니다.")
        return True
    
    st.warning(f"{filename} 파일에서 메시지를 찾을 수 없습니다.")
    return False

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file", type=['pdf','docx'], accept_multiple_files=True)

        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

        st.divider()
        
        model_name = st.selectbox("모델 선택", ("gpt-3.5-turbo", "gpt-4"), index=0)

        st.divider()
        
        st.subheader("채팅 기록 관리")
        chat_title = st.text_input("채팅 기록 제목 입력", "")
        
        if st.button("채팅 기록 저장"):
            save_chat_history(chat_title)

        st.subheader("채팅 기록 불러오기")
        # 저장된 채팅 기록 목록 가져오기
        if os.path.exists('chat_history'):
            files = [f for f in os.listdir('chat_history') if f.endswith('.txt')]
            if files:
                selected_file = st.selectbox("불러올 채팅 기록 선택", [""] + files)
                if st.button("선택한 기록 불러오기") and selected_file:
                    load_chat_history(selected_file)

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, model_name) 

        st.session_state.processComplete = True

        # 메인 채팅 UI 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 저장된 채팅 기록 표시
        with st.expander("저장된 채팅 기록", expanded=False):
            display_saved_chats()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
