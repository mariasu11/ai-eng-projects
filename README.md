# AI Engineering Projects Overview

This repository contains five hands-on projects completed over five weeks, followed by a capstone in week six. The first five projects build capability with large language models, retrieval, tool use, research workflows, and multimodality. Week six is a capstone where you design your own system, tool, or startup idea based on your learnings. The instructions below are generic and apply to all projects. Each project also includes additional instructions specific to that project.

Each week, a new project is added to the repo at a specific release date and time. The weekly release includes the notebook, data, and environment file.

## Quick start

You can run the projects either on **Google Colab** (no local setup required) or **locally** (using Conda environments for reproducibility).

### Option A: Run in Google Colab
1. Upload the notebook for the current week to Colab.
2. If needed, add your API tokens using `os.environ[...] = "value"`.
3. Ensure that any local file paths are adjusted for Colab.

### Option B: Run locally with Conda
Each project comes with an `environment.yml` file that specifies its dependencies. This ensures consistent environments.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).
2. Create and activate the environment from the provided YAML file:
   ```bash
   conda env create -f environment.yml
   conda activate <ENV_NAME>
   ```
   The environment name is set inside the YAML. You can change it if desired.
3. Launch Jupyter and open the notebook for the current week:
   ```bash
   jupyter notebook
   ```

**Recommendation:** Use Colab for projects 1 and 5, and local development for projects 2, 3, and 4.

## Accounts and keys you may need

The projects are designed so they do not require specific API keys or tokens by default. However, they are flexible, meaning you can switch to different LLMs, models, and systems. Depending on what you choose to experiment with, you may need to set up API keys or tokens from certain providers.  

Possible API keys you might need:
- `OPENAI_API_KEY` for OpenAI models  
- `ANTHROPIC_API_KEY` for Claude models  
- `GOOGLE_API_KEY` for Gemini models  
- `HUGGINGFACEHUB_API_TOKEN` for Hugging Face hosted models and datasets  
- `TAVILY_API_KEY` or `SERPAPI_API_KEY` for web search tools  
- `PINECONE_API_KEY`, or alternatives if using remote vector stores  

## Project expectations

- Projects are designed flexibly. They guide you step by step and provide the workflow. You will need to implement the sections marked with "your code here".  
- There are multiple ways to implement each section. Feel free to deviate from the provided template and experiment with different algorithms, models, and systems.  
- No submission is required. In the live deep-dive sessions, we will review each project in detail and show one possible implementation.  

## Troubleshooting

- Post questions in the corresponding Q/A space. You are also welcome to share your thoughts, opinions, and interesting findings in the same space.  

## Weekly projects

### Project 1: Build an LLM Playground
An introductory project to explore how prompts, tokenization, and decoding settings work in practice, building the foundation for effective use of large language models.

**Release Date**: October 4, 2025 · 10:00 AM PT

**Notebook Link:** [Here](https://github.com/bytebyteai/ai-eng-projects/blob/main/project_1/lm_playground.ipynb)  

**Learning objectives:**
- Tokenization of raw text into discrete tokens
- Basics of GPT-2 and Transformer architectures
- Loading pre-trained LLMs with Hugging Face
- Decoding strategies for text generation
- Completion vs. instruction-tuned models

### Project 2: Customer-Support Chatbot for an E-Commerce Store
A hands-on project to build a retrieval-based chatbot that answers customer questions for an imaginary e-commerce store.

**Release Date**: October 11, 2025 · 10:00 AM PT

**Notebook Link:** [Not Released Yet] 

**Learning objectives:**
- Ingest and chunk unstructured documents
- Create embeddings and index with FAISS
- Retrieve context and design prompts
- Run an open-weight LLM locally with Ollama
- Build a RAG (Retrieval-Augmented Generation) pipeline
- Package the chatbot in a minimal Streamlit UI

### Project 3: Ask-the-Web Agent
A project to create a simplified Perplexity-style agent that searches the web, reads content, and provides answers.

**Release Date**: October 18, 2025 · 10:00 AM PT

**Notebook Link:** [Not Released Yet] 

**Learning objectives:**
- Understand why tool calling is useful for LLMs
- Implement a loop to parse model calls and execute Python functions
- Use function schemas (docstrings and type hints) to scale across tools
- Apply LangChain for function calling, reasoning, and multi-step planning
- Combine Llama-3 7B Instruct with a web search tool to build an ask-the-web agent

### Project 4: Build a Deep Research System
A project focused on reasoning workflows, where you design a multi-step agent that plans, gathers evidence, and synthesizes findings.

**Release Date**: October 25, 2025 · 10:00 AM PT

**Notebook Link:** [Not Released Yet] 

**Learning objectives:**
- Apply inference-time scaling methods (zero-shot/few-shot CoT, self-consistency, sequential decoding, tree-of-thoughts)
- Gain intuition for training reasoning models with the STaR approach
- Build a deep-research agent that combines reasoning with live web search
- Extend deep-research into a multi-agent system

### Project 5: Build a Multimodal Agent
A project to build an agent that combines textual question answering with image and video generation capabilities within a unified system.

**Release Date**: November 1, 2025 · 10:00 AM PT

**Notebook Link:** [Not Released Yet] 

**Learning objectives:**
- Generate images from text using Stable Diffusion XL
- Create short clips with a text-to-video model
- Build a multimodal agent that handles questions and media requests
- Develop a simple Gradio UI to interact with the agent

### Week 6: Capstone Project

**Purpose:** Design and build your own system based on what you learned in weeks 1 to 5. This can be a product prototype, an internal tool, a research workflow, or the first step toward a startup idea. The hope is that some projects will continue after the cohort, using the connections and community built here.

**Demo date**: November 9, 2025 · 10:00 - 12:00 PM PT


## Reference docs and readings

The following documentation pages cover the core libraries and services used across projects:
- [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html): Manage isolated Python environments and dependencies with Conda  
- [Pip documentation](https://pip.pypa.io/en/stable/user_guide/): Install and manage Python packages with pip  
- [duckduckgo-search](https://pypi.org/project/duckduckgo-search/): Python library to query DuckDuckGo search results programmatically  
- [gradio](https://www.gradio.app/guides): Build quick interactive demos and UIs for machine learning models  
- [Streamlit documentation](https://docs.streamlit.io/): Build and deploy simple web apps for data and ML projects  
- [huggingface_hub](https://huggingface.co/docs/huggingface_hub/index): Access and share models, datasets, and spaces on Hugging Face Hub  
- [langchain](https://python.langchain.com/docs/get_started/introduction): Framework for building applications powered by LLMs with memory, tools, and chains  
- [numpy](https://numpy.org/doc/stable/): Core library for numerical computing and array operations in Python  
- [openai](https://platform.openai.com/docs): Official API docs for using OpenAI models like GPT and embeddings  
- [tiktoken](https://github.com/openai/tiktoken): Fast tokenizer library for OpenAI models, used for counting tokens  
- [torch](https://pytorch.org/docs/stable/index.html): PyTorch deep learning framework for training and running models  
- [transformers](https://huggingface.co/docs/transformers/index): Hugging Face library for using pre-trained LLMs and fine-tuning them  
- [llama-index](https://docs.llamaindex.ai/en/stable/): Data framework for connecting external data sources to LLMs  
- [chromadb](https://docs.trychroma.com/): Open-source vector database for storing and retrieving embeddings in RAG systems  



