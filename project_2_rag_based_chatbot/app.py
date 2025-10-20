import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

# --- Load your existing FAISS vectorstore and retriever ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load embeddings and vector store
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# --- Reuse your prompt ---
from langchain.prompts import PromptTemplate

SYSTEM_TEMPLATE = """
You are a Customer Support Chatbot. Use ONLY the <context> to answer.
If the answer is not in the context, reply exactly: "I'm not sure from the docs."

<context>
{context}
</context>

User question: {question}
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)

# --- Initialize model + chain ---
llm = Ollama(model="gemma3:1b", temperature=0.1)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# --- Streamlit UI ---
st.title("💬 Everstorm RAG Chatbot Demo")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.chat_input("Ask something about refunds, shipping, or support...")

if user_input:
    result = chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
    answer = result["answer"]

    # Update chat history
    st.session_state.chat_history.append((user_input, answer))

    # Display conversation
    for q, a in st.session_state.chat_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)
