# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: rag-chatbot
#     language: python
#     name: rag-chatbot
# ---

# %% [markdown]
# # Project 2: Customer‑Support Chatbot for an E-Commerce Store

# %% [markdown]
# Welcome! In this project, you'll build a **chatbot** that answers customer service questions about Everstorm Outfitters, an imaginary e-commerce store.
#
# Run each cell in order. Feel free to modify them as you go to better understand each tool and search the web or look online for documentation.

# %% [markdown]
# ## Learning Objectives  
# * **Ingest & chunk** unstructured docs  
# * **Embed** chunks and **index** with *FAISS*  
# * **Retrieve** context and **craft prompts**  
# * **Run** an open‑weight LLM locally with *Ollama*  
# * **Build** a Retrieval-Augmented Generation (RAG) chain
# * **Package** the chat loop in a minimal **Streamlit** web UI

# %% [markdown]
# ## Roadmap  
# We will build a RAG-based chatbot in **six** steps:
#
# 1. **Environment setup**
# 2. **Data preparation**  
#    a. Load source documents  
#    b. Chunk the text  
# 3. **Build a retriever**  
#    a. Generate embeddings  
#    b. Build a FAISS vector index  
# 4. **Build a generation engine**. Load the *Gemma3-1B* model through Ollama and run a sanity check.  
# 5. **Build a RAG**. Connect the system prompt, retriever, and LLM together. 
# 6. **(Optional) Streamlit UI**. Wrap everything in a simple web app so users can chat with the bot.
#

# %% [markdown]
# ## 1 - Environment setup
#
# We use conda to manage our project dependencies and ensure everyone has a consistent setup. Conda is an open-source package and environment manager that makes it easy to install libraries and switch between isolated environments. To learn more about conda, you can read: https://docs.conda.io/en/latest/
#
# Create and activate a clean *conda* environment and install the required packages. If you don't have conda installed, visit https://www.anaconda.com/docs/getting-started/miniconda/main.
#
#
# Open your terminal, navigate to the project folder where this notebook is located, and run the following commands.
#
# ```bash
# conda env create -f environment.yml && conda activate rag-chatbot
#
# # (Optional but recommended) Register this environment as a Jupyter kernel
# python -m ipykernel install --user --name=rag-chatbot --display-name "rag-chatbot"
# ```
# Once this is done, you can select “rag-chatbot” from the Kernel → Change Kernel menu in Jupyter or VS Code.
#
#
# > Behind the scenes:
# > * Conda reads `environment.yml`, solves all pinned dependencies, and builds an isolated environment named `rag-chatbot`.
# > * When it reaches the file’s "pip:" section, Conda automatically invokes pip to install any remaining Python-only packages so the whole stack be available for the project.
# > * Registering the kernel makes your new environment visible to Jupyter, so the notebook runs inside the same environment you just created.

# %% [markdown]
# Let's import required libraries and print a message if we're not **missing packages**.

# %%
# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

print("✅ Libraries imported! You're good to go!")

# %% [markdown]
# ## 2 - Data preparation
# The goal of this step is to turn all reference documents into small chunks of text that a retriever can index and search. These documents typically come from:
# * PDF files: local documents such as policies, user manuals, or guides.
# * Web pages (HTML): online documentation, blog posts, or help articles.
#
# In this step, we perform two actions:
# * **Ingesting**: load every PDF and collect the raw text in a list named `raw_docs`.
# * **Chunking**: split each document into small, overlapping chunks so later steps can match a user query to the most relevant passage.

# %% [markdown]
# ### 2.1 - Ingest source documents

# %% [markdown]
# We can use different libraries to load and process PDFs. A quick web search will show several options, each with its own strengths. In this case, we’ll use PyPDFLoader from LangChain, which makes it easy to extract text from PDF files for downstream processing. To learn more about how to use it, refer to: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
#
# Use **PyPDFLoader** to load every PDF whose filename matches `data/Everstorm_*.pdf` and collect all pages in a list called `raw_docs`. The content of these PDFs is synthetically generated for educational purposes.

# %%
pdf_paths = glob.glob("data/Everstorm_*.pdf")
raw_docs = []

for path in pdf_paths:
    loader = PyPDFLoader(path)
    raw_docs.extend(loader.load())

print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")

# %% [markdown]
# ### (Optional) 2.1 - Load web pages
# You can also pull content straight from the web. Various libraries support reading and parsing web pages directly into text, which is useful for building custom knowledge bases. One example is **UnstructuredURLLoader** from LangChain, which can extract readable content from raw HTML pages and return them in a structured format. To learn more, see: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.url.UnstructuredURLLoader.html
#
# To practice, load each HTML page below and store the results in a list called `raw_docs`. We’ve included a few sample URLs, but you can replace them with any links you prefer.
#
# For robustness, add an offline fallback in case a URL fails. In real projects, we typically cache fetched pages to disk, handle rate limits, and track fetch timestamps so content can be refreshed periodically without relying on live network calls during development. For this project, we don’t have offline HTML copies available, but you can still practice by loading any PDFs from the data/ folder using PyPDFLoader and appending them to raw_docs.

# %%
URLS = [
    # --- BigCommerce – shipping & refunds ---
    "https://developer.bigcommerce.com/docs/store-operations/shipping",
    "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
    # --- Stripe – disputes & chargebacks ---
    # "https://docs.stripe.com/disputes",  
    # --- WooCommerce – REST API reference ---
    # "https://woocommerce.github.io/woocommerce-rest-api-docs/v3.html",
]

try:
    loader = UnstructuredURLLoader(urls=URLS)
    raw_docs = loader.load()
    print(f"Fetched {len(raw_docs)} documents from the web.")
except Exception as e:
    print("⚠️  Web fetch failed, using offline copies:", e)
    raw_docs = []
    pdf_paths = glob.glob("data/Everstorm_*.pdf")
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        raw_docs.extend(loader.load())
    print(f"Loaded {len(raw_docs)} offline documents.")

# %%
# 1) Append PDFs to existing raw_docs
pdf_paths = glob.glob("data/Everstorm_*.pdf")
for path in pdf_paths:
    loader = PyPDFLoader(path)
    raw_docs.extend(loader.load())

print(f"Now have {len(raw_docs)} total documents (web + PDF pages).")

# %% [markdown]
# ### 2.2 - Chunk the text

# %% [markdown]
# Long documents won’t work well directly with most LLMs. They can easily exceed the model’s context window, making it impossible for the model to read or reason over the full text at once. Even if they fit, processing long inputs can be inefficient and lead to weaker retrieval results.
#
# To handle this, we split large documents into smaller, overlapping chunks. Several libraries can help with text splitting, each designed to preserve structure or balance chunk size. A popular choice is `RecursiveCharacterTextSplitter` from LangChain, which splits text intelligently while keeping paragraph or sentence boundaries intact. To familiarize youself with the library, visit: https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
#
# In this project, we’ll split each document into chunks of roughly 300 tokens with a 30-token overlap using `RecursiveCharacterTextSplitter`. This overlap helps maintain continuity across chunks while ensuring each piece stays small enough for embedding and retrieval.

# %%
chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(raw_docs)
print(f"✅ {len(chunks)} chunks ready for embedding")

# %% [markdown]
# ## 3 -Build a retriever
#
# A *retriever* lets the RAG pipeline efficiently look up small, relevant pieces of context at query‑time. This step has two parts:
# 1. **Load a model to generate embeddings**: convert each text chunk from the reference documents into a fixed‑length vector that captures its semantic meaning.  
# 2. **Build vector database**: store these embeddings in a vector database.
#

# %% [markdown]
# ### 3.1 - Load a model to generate embeddings

# %% [markdown]
# The goal of this step is to convert each document chunk into a numerical vector (an embedding) that captures its semantic meaning. These embeddings allow our retriever to find and compare similar pieces of text efficiently.
#
# There are models trained specifically for this purpose, called embedding models. One popular example is OpenAI’s `text-embedding-3-small`, which produces high-quality embeddings that work well for retrieval and semantic search.
#
# If you prefer running everything locally, you can use smaller open-source models such as `gte-small` (77 M parameters). These local models load quickly, don’t require internet access, and are ideal for experimentation or environments without API access. However, they’re typically less powerful than hosted models.
#
# Alternatively, you can connect to an API service to access stronger models like OpenAI’s. These require setting an API key (for example, OPENAI_API_KEY) in your environment. OpenAI allows you to create a free account and sometimes offers limited trial credits for new users, but ongoing access requires a billing setup. 
#
# In this exercise, we’ll stick to the smaller gte-small model for simplicity and reproducibility. We'll use our imported `SentenceTransformerEmbeddings` library to load the model and use it to embed queries. To learn more about lagnchain's embedding support, visit: https://python.langchain.com/docs/integrations/text_embedding/

# %%
embedding_vector = []

# Embed the sentence "Hello world! and store it in an embedding_vector.
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_vector = embedding_model.embed_query("Hello world!")

print(len(embedding_vector))

# %% [markdown]
# ### 3.2 - Build a vector database
#
# Once we have embeddings, we need a way to store and search them efficiently. A simple list wouldn’t scale well, especially when we have thousands of chunks and need to quickly find the most relevant ones.
#
# To solve this, we use **FAISS**, an open-source similarity search library developed by Meta. FAISS is optimized for fast nearest-neighbor search in high-dimensional spaces, making it ideal for tasks like semantic retrieval and recommendation. It’s strongly encouraged to visit their quickstart guide to understand how FAISS works and how to use it effectively: https://github.com/facebookresearch/faiss/wiki/getting-started
#
# In this step, we’ll feed all our document embeddings into FAISS, which builds an in-memory vector index. This index allows us to efficiently query for the *k* most similar chunks to any given question.
#
# During inference, we’ll use this index to retrieve the top-k relevant chunks and pass them to the LLM as context, enabling it to answer questions grounded in our documents.
#
#

# %%
# Expected steps:
    # 1. Build the FAISS index from the list of document chunks and their embeddings.
    # 2. Create a retriever object with a suitable k value (e.g., 8).
    # 3. Save the vector store locally (e.g., under "faiss_index").
    # 4. Print a short confirmation showing how many embeddings were stored.

# 1. Load embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Build FAISS index from the list of document chunks
vectordb = FAISS.from_documents(chunks, embedding_model)

# 3. (Optional) Create a retriever for top-k searches
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# 4. Save the FAISS vector store locally
vectordb.save_local("faiss_index")

print("✅ Vector store with", vectordb.index.ntotal, "embeddings")

# %% [markdown]
# ## 4 - Build the generation engine
# At the core of any RAG system lies an **LLM**. The retriever finds relevant information, and the LLM uses that information to generate coherent, context-aware responses.
#
# In this project, we’ll use **Gemma 3* (1B), a small but capable open-weight model, and run it entirely on your local machine using Ollama. This means you won’t need API keys or internet access to generate responses once the model is downloaded.
#
# **What is Ollama?**
#
# Ollama is a lightweight runtime for managing and serving open-weight LLMs locally. It provides:
# * A simple REST API running at localhost:11434, so your code can interact with the model via standard HTTP calls.
# * A model registry and command-line tool** to pull, run, and manage models easily.
# * Support for a wide variety of models (e.g., Gemma, Llama, Mistral, Phi, etc.), making it ideal for experimentation.
#
# To learn more about Ollama, visit: https://github.com/ollama/ollama. You can browse all supported models and their sizes here: https://ollama.com/library
#
#
# ### 4.1 - Install `ollama` and serve `gemma3`
#
# Follow these steps to set up Ollama and start the model server:
#
# **1 - Install**
# ```bash
# # macOS (Homebrew)
# brew install ollama
# # Linux
# curl -fsSL https://ollama.com/install.sh | sh
# ```
#
# If you’re on Windows, install using the official installer from https://ollama.com/download.
#
# **2 - Start the Ollama server (keep this terminal open)**
# ```bash
# ollama serve
# ```
# This command launches a local server at http://localhost:11434, which will stay running in the background.
#
#
# **3 - Pull the Gemma mode (or the model of your choice) in a new terminal**
# ```bash
# ollama pull gemma3:1b
# ```
#
# This downloads the 1B version of Gemma 3, a compact model suitable for running on most modern laptops. Once downloaded, Ollama will automatically handle model loading and caching.
#
#
# After this setup, your system is ready to generate responses locally using the Gemma model through the Ollama API.
#

# %% [markdown]
# ### 4.2 - Test an LLM with a random prompt (Sanity check)
#

# %%
# Expected steps:
    # 1. Initialize the model (for example, gemma3:1b) with a low temperature such as 0.1 for more factual outputs.
    # 2. Use llm.invoke() with a short test prompt and print the response to verify that the model runs successfully.

llm = Ollama(model="gemma3:1b", temperature=0.1)
# Test prompt to verify it’s working
response = llm.invoke("Hello! Can you briefly introduce yourself?")
print(response)

# %% [markdown]
# ## Build a RAG

# %% [markdown]
# ### 5.1 - Define a system prompt

# %% [markdown]
# At this stage, we need to tell the model how to behave when generating answers. The **system prompt** acts as the model’s rulebook. It should clearly instruct the model to answer only using the retrieved context and to admit when it doesn’t know the answer. This helps prevent hallucination and keeps the responses grounded in the provided documents.
#
# In general, a good RAG prompt emphasizes three things: stay within context, stay factual, and stay concise. This is important because RAG works by grounding the LLM in retrieved text. If the prompt is vague, the model may invent details. A precise system prompt reduces hallucinations and keeps answers aligned with your corpus.

# %%
SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: {{source}}] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

# %% [markdown]
# ### 5.2 Create a RAG chain
# Now that we have a retriever, a prompt, and a language model, we can connect them into a single RAG pipeline. The retriever finds the most relevant chunks from our vector index, the prompt injects those chunks into the system message, and the LLM uses that context to produce the final answer. (retriever → prompt → model)
#
# This connection is handled through LangChain’s `ConversationalRetrievalChain`, which combines retrieval and generation. To familiarize yourself with the library, visit: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html

# %%
# Expected steps:
    # 1. Create a PromptTemplate that uses the SYSTEM_TEMPLATE you defined earlier, with input variables for "context" and "question".
    # 2. Initialize your LLM using Ollama with the gemma3:1b model and a low temperature (e.g., 0.1) for reliable, grounded responses.
    # 3. Build a ConversationalRetrievalChain by combining the LLM, the retriever, and your custom prompt and name it "chain".

prompt = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)
llm = Ollama(model="gemma3:1b", temperature=0.1)
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, combine_docs_chain_kwargs={"prompt": prompt})


# %% [markdown]
# When you ask a question, the retriever pulls the top few relevant text chunks, the model reads them through the system prompt, and then it generates an answer based on that context.
#
# This structure makes the system transparent and easy to debug. You can inspect what text was retrieved, tune parameters like k, and experiment with different prompts to see how they affect the output quality.
#

# %% [markdown]
# ### 5.3 - Validate the RAG chain

# %% [markdown]
# We run a few questions to make sure everything behaves as expecte. Experiment by adding you own questions.

# %%
test_questions = [
    "If I'm not happy with my purchase, what is your refund policy and how do I start a return?",
    "How long will delivery take for a standard order, and where can I track my package once it ships?",
    "What's the quickest way to contact your support team, and what are your operating hours?",
]

# Expected steps:
    # 1. Initialize an empty chat_history list.
    # 2. Loop through test_questions, pass each question and the current chat history to the chain, and append the new answer.
    # 3. Print each question and the LLM's response to verify it’s working correctly.

chat_history = []

for question in test_questions:
    result = chain.invoke({"question": question, "chat_history": chat_history})
    answer = result["answer"]

    # Append to chat history to keep the conversation context
    chat_history.append((question, answer))

    print(f"🧠 Question: {question}")
    print(f"💬 Answer: {answer}\n")

# %% [markdown]
# ### 6 - Build the Streamlit UI (optional)

# %% [markdown]
# The goal here is to create a tiny demo so you can interact with your RAG system. The focus is not on UI design. We will build a very small interface only to demonstrate the end-to-end flow.
#
# There are many ways to make a UI. Some frameworks are powerful but take longer to set up, while others are simple and good for quick experiments. Streamlit is a common choice for fast prototyping because it lets you make a usable interface with only a few lines of Python. If you want to learn the basics, see the Streamlit Quickstart: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart
#
# This step is optional. If it is not useful for your work, you can skip it. We will also complete this part together during the live session.
#
# In this cell, we write a minimal **`app.py`** that starts a simple chat UI and calls your RAG chain.

# %%
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


# %% [markdown]
# Run `streamlit run app.py` from your terminal.

# %% [markdown]
# ## 🎉 Congratulations!
#
# You’ve just built, tested, and demoed a fully working **customer-support chatbot**.  
# In one project you:
#
# * **Prepared policy docs**: loaded and chunked them for fast retrieval.  
# * **Built a vector store**: created a FAISS index with text embeddings.  
# * **Plugged in an LLM**: wrapped Gemma3 with LangChain and a prompt-aware RAG chain.  
# * **Validated end-to-end**: answered refund, shipping, and contact questions with confidence.  
#
# Swap in new documents, tweak the prompt, and your store’s customers get instant, accurate answers.
#
# 👏 **Great job!** Take a moment to celebrate. The skills you used here power most RAG-based chatbots you see everywhere.
#

# %% [markdown]
#
