import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import os

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Helper function to extract video ID from URL
def extract_video_id(url):
    pattern = r"(?:youtu\.be/|v=)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Streamlit UI
st.title("Talk to YouTube 📹")
st.markdown("Paste a YouTube video link and ask questions about its content!")

# Video input
video_url = st.text_input("Paste YouTube Video Link", "https://youtu.be/uaTfTikNe9w")
if video_url and video_url != st.session_state.video_id:
    video_id = extract_video_id(video_url)
    if video_id:
        st.session_state.video_id = video_id
        with st.spinner("Fetching transcript..."):
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(i["text"] for i in transcript_list)
                st.success("Transcript loaded!")

                # Process transcript
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunk = splitter.create_documents([transcript])
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                index_path = f"faiss_index_{video_id}"
                if os.path.exists(index_path):
                    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                else:
                    vector_store = FAISS.from_documents(chunk, embeddings)
                    vector_store.save_local(index_path)
                st.session_state.vector_store = vector_store
            except Exception as e:
                st.error(f"Could not load transcript: {e}")
                st.session_state.vector_store = None
    else:
        st.error("Invalid YouTube URL. Please try again.")

# Chat interface
if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = PromptTemplate(
        template="""
          You're a friendly assistant who loves explaining YouTube videos!
          Answer the question based on the transcript context in a clear, engaging way.
          If the transcript doesn't have enough info, say, 'Hmm, I couldn't find that in the video. Want to ask something else?'

          {context}
          Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about the video...")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.spinner("Thinking..."):
            try:
                result = main_chain.invoke(user_question)
                with st.chat_message("assistant"):
                    st.write(result)
                st.session_state.chat_history.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Oops, something went wrong: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I hit a snag. Try asking again!"})
else:
    st.info("Enter a valid YouTube video link to start chatting!")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()