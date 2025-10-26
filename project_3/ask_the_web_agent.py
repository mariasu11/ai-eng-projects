# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: web_agent
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project 3: **Ask‑the‑Web Agent**
#
# Welcome to Project 3! In this project, you will learn how to use tool‑calling LLMs, extend them with custom tools, and build a simplified *Perplexity‑style* agent that answers questions by searching the web.

# %% [markdown]
# ## Learning Objectives  
# * Understand why tool calling is useful and how LLMs can invoke external tools.
# * Implement a minimal loop that parses the LLM's output and executes a Python function.
# * See how *function schemas* (docstrings and type hints) let us scale to many tools.
# * Use **LangChain** to get function‑calling capability for free (ReAct reasoning, memory, multi‑step planning).
# * Combine LLM with a web‑search tool to build a simple ask‑the‑web agent.

# %% [markdown]
# ## Roadmap
# 1. Environment setup
# 2. Write simple tools and connect them to an LLM
# 3. Standardize tool calling by writing `to_schema`
# 4. Use LangChain to augment an LLM with your tools
# 5. Build a Perplexity‑style web‑search agent
# 6. (Optional) A minimal backend and frontend UI

# %% [markdown] vscode={"languageId": "shellscript"}
# # 1- Environment setup
#
# ## 1.1- Conda environment
#
# Before we start coding, you need a reproducible setup. Open a terminal in the same directory as this notebook and run:
#
# ```bash
# # Create and activate the conda environment
# conda env create -f environment.yml && conda activate web_agent
#
# # Register this environment as a Jupyter kernel
# python -m ipykernel install --user --name=web_agent --display-name "web_agent"
# ```
# Once this is done, you can select “web_agent” from the Kernel → Change Kernel menu in Jupyter or VS Code.
#
#
# > Behind the scenes:
# > * Conda reads `environment.yml`, resolves the pinned dependencies, creates an isolated environment named `web_agent`, and activates it.
# > * `ollama pull` downloads the model so you can run it locally without API calls.
#
#
# ## 1.2 Ollama setup
#
# In this project, we start with `gemma3-1B` because it is lightweight and runs on most machines. You can try other smaller or larger LLMs such as `mistral:7b`, `phi3:mini`, or `llama3.2:1b` to compare performance. Explore available models here: https://ollama.com/library
#
# ```bash
# ollama pull gemma3:1b
# ```
#
# `ollama pull` downloads the model so you can run it locally without API calls.
#

# %% [markdown]
# ## 2- Tool Calling
#
# LLMs are strong at answering questions, but they cannot directly access external data such as live web results, APIs, or computations. In real applications, agents rarely rely only on their internal knowledge. They need to query APIs, retrieve data, or perform calculations to stay accurate and useful. Tool calling bridges this gap by allowing the LLM to request actions from the outside world.
#
#
# We describe each tool’s interface in the model’s prompt, defining what it does and what arguments it expects. When the model decides that a tool is needed, it emits a structured output like: `TOOL_CALL: {"name": "get_current_weather", "args": {"city": "San Francisco"}}`. Your code will detect this output, execute the corresponding function, and feed the result back to the LLM so the conversation continues.
#
# In this section, you will implement a simple `get_current_weather` function and teach the `gemma3` model how to use it when required in four steps:
# 1. Implement the tool
# 2. Create the instructions for the LLM
# 3. Call the LLM with the prompt
# 4. Parse the LLM output and call the tool

# %%
from openai import OpenAI

client = OpenAI(api_key = "ollama", base_url = "http://localhost:11434/v1")


# %%
# ---------------------------------------------------------
# Step 1: Implement the tool
# ---------------------------------------------------------
# Your goal: give the model a way to access weather information.
# You can either:
#   (a) Call a real weather API (for example, OpenWeatherMap), or
#   (b) Create a dummy function that returns a fixed response (e.g., "It is 23°C and sunny in San Francisco.")
#
# Requirements:
#   • The function should be named `get_current_weather`
#   • It should take two arguments:
#         - city: str
#         - unit: str = "celsius"
#   • Return a short, human-readable sentence describing the weather.
#
# Example expected behavior:
#   get_current_weather("San Francisco") → "It is 23°C and sunny in San Francisco."
#

def get_current_weather(city: str, unit: str = "celsius") -> str:
    print(f"[TOOL] get_current_weather(city={city!r}, unit={unit!r})")  # tracer so you *see* it run
    temperature = 23 if unit == "celsius" else 73.4
    return f"It is {temperature}°{ 'C' if unit == 'celsius' else 'F' } and sunny in {city}."



# %%
# ---------------------------------------------------------
# Step 2: Create the prompt for the LLM to call tools
# ---------------------------------------------------------
# Goal:
#   Build the system and user prompts that instruct the model when and how
#   to use your tool (`get_current_weather`).
#
# What to include:
#   • A SYSTEM_PROMPT that tells the model about the tool use and describe the tool
#   • A USER_QUESTION with a user query that should trigger the tool.
#       Example: "What is the weather in San Diego today?"

# Try experimenting with different system and user prompts
# ---------------------------------------------------------

# ---------------------------------------------------------
# Step 2: Create the prompt for the LLM to call tools
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are gemma3, a helpful assistant with tool-calling abilities.

## Tool Calling Protocol
- When you need external information, emit a single line in this exact format:
  TOOL_CALL: {"name": "<tool_name>", "args": { ... }}
- Do NOT add any other text on that line.
- After the host executes the tool, it will feed you the result as:
  TOOL_RESULT: <string>
- When you receive TOOL_RESULT, respond to the user with a clear, human-friendly answer.
- Only call tools when needed (e.g., factual, up-to-date, or user-specific data).
- If the user’s request is ambiguous (e.g., missing city), ask a brief clarifying question instead of guessing.

## Available Tools
- get_current_weather(city: str, unit: str = "celsius") -> str
  • Fetches current weather and returns a short sentence.
  • unit must be one of: "celsius" | "fahrenheit"
  • If the user asks for "F", "°F", or "fahrenheit", use "fahrenheit".
  • Default to "celsius" if unit is not specified.

## Decision Rules
- If the user asks about current weather for a city, call get_current_weather.
- If multiple cities are requested, issue one TOOL_CALL per city (separately).
- Do not make up weather; always use the tool for live/accurate data.
- If the city is unknown or missing, ask for the city before calling the tool.
- If the tool returns an error, apologize briefly and ask for a correction.

## Examples
User: What's the weather in San Francisco?
Assistant: TOOL_CALL: {"name": "get_current_weather", "args": {"city": "San Francisco", "unit": "celsius"}}

(Host executes and returns)
TOOL_RESULT: It is 23°C and sunny in San Francisco.
Assistant: It’s 23°C and sunny in San Francisco.

User: Weather in Austin in °F?
Assistant: TOOL_CALL: {"name": "get_current_weather", "args": {"city": "Austin", "unit": "fahrenheit"}}

User: Tell me about photosynthesis.
Assistant: Photosynthesis is the process by which plants convert light energy into chemical energy...
"""

# A user question that should trigger tool use
USER_QUESTION = "What is the weather in San Diego today in Fahrenheit?"


# %% [markdown]
# Now that you have defined a tool and shown the model how to use it, the next step is to call the LLM using your prompt.
#
# Start the **Ollama** server in a terminal with `ollama serve`. This launches a local API endpoint that listens for LLM requests. Once the server is running, return to the notebook and in the next cell send a query to the model.
#

# %%
# ---------------------------------------------------------
# Step 3: Call the LLM with your prompt
# ---------------------------------------------------------
# Task:
#   Send SYSTEM_PROMPT + USER_QUESTION to the model.
#
# Steps:
#   1. Use the Ollama client to create a chat completion. 
#       - You may find some examples here: https://platform.openai.com/docs/api-reference/chat/create
#       - If you are unsure, search the web for "client.chat.completions.create"
#   2. Print the raw response.
#
# Expected:
#   The model should return something like:
#   TOOL_CALL: {"name": "get_current_weather", "args": {"city": "San Diego"}}
# ---------------------------------------------------------


response = client.chat.completions.create(
    model="gemma3:1b",   # your local Ollama model name
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_QUESTION},
    ]
)

# Print the raw response to inspect the tool call
print(response)

# %%
# ---------------------------------------------------------
# Step 4: Parse the LLM output and call the tool
# ---------------------------------------------------------
# Task:
#   Detect when the model requests a tool, extract its name and arguments,
#   and execute the corresponding function.
#
# Steps:
#   1. Search for the text pattern "TOOL_CALL:{...}" in the model output.
#   2. Parse the JSON inside it to get the tool name and args.
#   3. Call the matching function (e.g., get_current_weather).
#
# Expected:
#   You should see a line like:
#       Calling tool `get_current_weather` with args {'city': 'San Diego'}
#       Result: It is 23°C and sunny in San Diego.
# ---------------------------------------------------------

import re, json

llm_text = response.choices[0].message.content

m = re.search(r'TOOL_CALL:\s*(\{.*\})', llm_text, re.DOTALL)
if not m:
    print("No tool call found. Model said:\n", llm_text)
else:
    try:
        payload = json.loads(m.group(1))
        name = payload.get("name")
        args = payload.get("args", {}) or {}
        tools = {"get_current_weather": get_current_weather}

        if name not in tools:
            print(f"Unknown tool requested: {name}")
        else:
            print(f"Calling tool `{name}` with args {args}")
            result = tools[name](**args)
            print("Result:", result)

          #  (Optional) Feed back to the model in a new turn:
            followup = client.chat.completions.create(
                model="gemma3:1b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_QUESTION},
                    {"role": "assistant", "content": llm_text},
                    {"role": "user", "content": f"TOOL_RESULT: {result}"}
                ]
            )
            print(followup.choices[0].message.content)

    except json.JSONDecodeError as e:
        print("Failed to parse TOOL_CALL JSON:", e)


# %% [markdown]
# # 3- Standadize tool calling
#
# So far, we handled tool calling manually by writing one regex and one hard-coded function. This approach does not scale if we want to add more tools. Adding more tools would mean more `if/else` blocks and manual edits to the `TOOL_SPEC` prompt.
#
# To make the system flexible, we can standardize tool definitions by automatically reading each function’s signature, converting it to a JSON schema, and passing that schema to the LLM. This way, the LLM can dynamically understand which tools exist and how to call them without requiring manual updates to prompts or conditional logic.
#
# Next, you will implement a small helper that extracts metadata from functions and builds a schema for each tool.

# %%
# ---------------------------------------------------------
# Generate a JSON schema for a tool automatically
# ---------------------------------------------------------
#
# Steps:
#   1. Use `inspect.signature` to get function parameters.
#   2. For each argument, record its name, type, and description.
#   3. Build a schema containing:
#   4. Test your helper on `get_current_weather` and print the result.
#
# Expected:
#   A dictionary describing the tool (its name, args, and types).
# ---------------------------------------------------------

from pprint import pprint
import inspect


def to_schema(fn):
    sig = inspect.signature(fn)
    schema = {
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    for name, param in sig.parameters.items():
        typ = param.annotation.__name__ if param.annotation != inspect._empty else "string"
        entry = {"type": typ.lower() if typ.lower() in ["string","number","integer","boolean"] else "string"}
        if param.default != inspect._empty:
            entry["default"] = param.default
        else:
            schema["parameters"]["required"].append(name)
        schema["parameters"]["properties"][name] = entry
    return schema

tool_schema = to_schema(get_current_weather)
pprint(tool_schema)

# %%
# ---------------------------------------------------------
# Provide the tool schema to the model
# ---------------------------------------------------------
# Goal:
#   Give the model a "menu" of available tools so it can choose
#   which one to call based on the user’s question.
#
# Steps:
#   1. Add an extra system message (e.g., name="tool_spec")
#      containing the JSON schema(s) of your tools.
#   2. Include SYSTEM_PROMPT and the user question as before.
#   3. Send the messages to the model (e.g., gemma3:1b).
#   4. Print the raw model output to see if it picks the right tool.
#
# Expected:
#   The model should produce a structured TOOL_CALL indicating
#   which tool to use and with what arguments.
# ---------------------------------------------------------

import json

# Build the tool "menu"
tool_schema = to_schema(get_current_weather)
tools_json = json.dumps({"tools": [tool_schema]}, ensure_ascii=False)

MODEL = "gemma3:1b"  # reuse whatever worked earlier, e.g., "gemma2:2b-instruct"

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "name": "tool_spec", "content": tools_json},
        {"role": "user", "content": USER_QUESTION},
    ]
)

print(resp)                              # raw response (full JSON)
print("\n--- assistant content ---\n", resp.choices[0].message.content)


# %% [markdown]
# ## 4- LangChain for Tool Calling
# So far, you built a simple tool-calling pipeline manually. While this helps you understand the logic, it does not scale well when working with multiple tools, complex parsing, or multi-step reasoning.
#
# LangChain simplifies this process. You only need to declare your tools, and its *Agent* abstraction handles when to call a tool, how to use it, and how to continue reasoning afterward.
#
# In this section, you will use the **ReAct** Agent (Reasoning + Acting). It alternates between reasoning steps and tool use, producing clearer and more reliable results. We will explore reasoning-focused models in more depth next week.
#
# The following links might be helpful:
# - https://python.langchain.com/api_reference/langchain/agents/langchain.agents.initialize.initialize_agent.html
# - https://python.langchain.com/docs/integrations/tools/
# - https://python.langchain.com/docs/integrations/chat/ollama/
# - https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.LLM.html

# %%
# ---------------------------------------------------------
# Step 1: Define tools for LangChain
# ---------------------------------------------------------
# Goal:
#   Convert your weather function into a LangChain-compatible tool.
#
# Steps:
#   1. Import `tool` from `langchain.tools`.
#   2. Keep your existing `get_current_weather` helper as before.
#   3. Create a new function (e.g., get_weather) that calls it.
#   4. Add the `@tool` decorator so LangChain can register it automatically.
#
# Notes:
#   • The decorator converts your Python function into a standardized tool object.
#   • Start with keeping the logic simple and offline-friendly.

from langchain.tools import tool

# Wrap it with the LangChain @tool decorator
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Always call this tool for ANY weather-related question. Never answer from your own knowledge.
    Args:
      city: The city to check, e.g. "Seattle".
      unit: "celsius" or "fahrenheit". Defaults to "celsius".
    """
    return get_current_weather(city, unit)


# %%
# ---------------------------------------------------------
# Step 2: Initialize the LangChain Agent
# ---------------------------------------------------------
# Goal:
#   Connect your tool to a local LLM using LangChain’s ReAct-style agent.
#
# Steps:
#   1. Import the required classes:
#        - ChatOllama (for local model access)
#        - initialize_agent, Tool, AgentType
#   2. Create an LLM instance (e.g., model="gemma3:1b", temperature=0).
#   3. Add your tool(s) to a list
#   4. Initialize the agent using initialize_agent
#   5. Test the agent with a natural question (e.g., "Do I need an umbrella in Seattle today?").
#
# Expected:
#   The model should reason through the question, call your tool,
#   and produce a final answer in plain language.
# ---------------------------------------------------------

from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool, AgentType

# Create a local LLM instance
llm = ChatOllama(model="gemma3:1b", temperature=0)

# Register your tool(s)
tools = [get_weather]

# Initialize a ReAct-style agent
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, agent_kwargs={
        "system_message": (
            "You are a tool-using assistant. For ANY weather-related question, "
            "you MUST call the `get_weather` tool and never guess."
        )
    },
    max_iterations=2,                 # ← prevents long loops
    early_stopping_method="generate", # ← forces a final answer
    verbose=True)

# Test the agent
#agent.run("Do I need an umbrella in Seattle today?")
agent.run("Do I need an umbrella in Seattle today?")



# %% [markdown]
# ### What just happened?
#
# The console log displays the **Thought → Action → Observation → …** loop until the agent produces its final answer. Because `verbose=True`, LangChain prints each intermediate reasoning step.
#
# If you want to add more tools, simply append them to the tools list. LangChain will handle argument validation, schema generation, and tool-calling logic automatically.

# %%
import sys, subprocess

# Install ddgs with its deps + HTTP/2 extras
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ddgs", "httpx[http2]"])
# Just in case, ensure h2 is present explicitly
subprocess.check_call([sys.executable, "-m", "pip", "install", "h2"])

# Sanity check
import ddgs
from ddgs import DDGS
print("ddgs version:", ddgs.__version__)


# %% [markdown]
#
# ## 5- Perplexity‑Style Web Search
# Agents become much more powerful when they can look up real information on the web instead of relying only on their internal knowledge.
#
# In this section, you will combine everything you have learned to build a simple Ask-the-Web Agent. You will integrate a web search tool (DuckDuckGo) and make it available to the agent using the same tool-calling approach as before.
#
# This will let the model retrieve fresh results, reason over them, and generate an informed answer—similar to how Perplexity works.
#
# You may find some examples from the following links:
# - https://pypi.org/project/duckduckgo-search/

# %%
# ---------------------------------------------------------
# Step 1: Add a web search tool
# ---------------------------------------------------------
# Goal:
#   Create a tool that lets the agent search the web and return results.
#
# Steps:
#   1. Use DuckDuckGo for quick, open web searches.
#   2. Write a helper function (e.g., search_web) that:
#        • Takes a query string
#        • Uses DDGS to fetch top results (titles + URLs)
#        • Returns them as a formatted string
#   3. Wrap it with the @tool decorator to make it available to LangChain.


# pip install ddgs
from ddgs import DDGS
from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator



class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query text")

    @model_validator(mode="before")
    def lift_query(cls, data):
        if isinstance(data, dict) and "query" not in data:
            for k in ("title", "description", "text", "q"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    data["query"] = v
                    break
        return data

@tool("search_web", args_schema=SearchArgs, return_direct=True)
def search_web(query: str) -> str:
    """Search DuckDuckGo and return the top 5 *recent* results (title — URL)."""
    with DDGS() as ddg:
        # Try text backends first
        results = []
        for backend in ("lite", "html"):
            hits = ddg.text(
                query,
                max_results=5,
                region="us-en",
                safesearch="moderate",
                timelimit="w",   # past week
                backend=backend
            )
            if hits:
                results = hits
                break
        # Fallback: news vertical
        if not results:
            results = ddg.news(
                query,
                max_results=5,
                region="us-en",
                safesearch="moderate",
                timelimit="w"
            ) or []

    if not results:
        return ("No results found. Try a more specific query, e.g., "
                "San Francisco events this week site:eventbrite.com OR site:sf.funcheap.com OR site:sftravel.com")

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title") or r.get("body") or "Untitled"
        url = r.get("href") or r.get("url") or ""
        lines.append(f"{i}. {title} — {url}")
    return "\n".join(lines)



# %% [markdown]
#
# Let’s see the agent's output in action with a real example.
#

# %%

# ---------------------------------------------------------
# Step 2: Initialize the web-search agent
# ---------------------------------------------------------
# Goal:
#   Connect your `web_search` tool to a language model
#   so the agent can search and reason over real data.
#
# Steps:
#   1. Import `initialize_agent` and `AgentType`.
#   2. Create an LLM (e.g., ChatOllama).
#   3. Add your `web_search` tool to the tools list.
#   4. Initialize the agent using: initialize_agent
#   5. Keep `verbose=True` to observe reasoning steps.
#
# Expected:
#   The agent should be ready to accept user queries
#   and use your web search tool when needed.
# ---------------------------------------------------------
from langchain.agents import initialize_agent, AgentType
#from langchain.llms import OpenAI

from langchain_community.chat_models import ChatOllama

# 1. Create a local LLM instance
llm = ChatOllama(model="mistral:7b-instruct", temperature=0)

# 2. Register your tool
tools = [search_web]

# 3. Initialize the web-search agent
web_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=2,                  # avoids loops
    early_stopping_method="generate",  # force a final answer if looping
)




# %%
print([t.name for t in web_agent.tools])  # should include 'search_web'


# %% [markdown]
#
# Let’s see the agent's output in action with a real example.
#

# %%

# ---------------------------------------------------------
# Step 3: Test your Ask-the-Web agent
# ---------------------------------------------------------
# Goal:
#   Verify that the agent can search the web and return
#   a summarized answer based on real results.
#
# Steps:
#   1. Ask a natural question that requires live information,
#      for example: "What are the current events in San Francisco this week?"
#   2. Call agent.
#
# Expected:
#   The agent should call `web_search`, retrieve results,
#   and generate a short summary response.
# ---------------------------------------------------------

# Ask a question that needs current info
#query = "What are the current events in San Francisco this week?"

# Run the agent (it should trigger the web search tool)
#response = web_agent.run(query)

# Print the summarized result
#print(response)


# result = web_agent.invoke({"input": "What are the current events in San Francisco this week?"})
# print(result["output"])

# # (optional) inspect tool calls
# for action, obs in result["intermediate_steps"]:
#     print("Action:", action)
#     print("Observation:", (obs[:200] + "...") if isinstance(obs, str) and len(obs) > 200 else obs)

result = web_agent.invoke({"input": "What are the current events in San Francisco this week?"})
print("\n=== OUTPUT ===")
print(result["output"] if isinstance(result, dict) and "output" in result else result)



# %% [markdown]
#
# ## 6- A minimal UI
# This project includes a simple **React** front end that sends the user’s question to a FastAPI back end and streams the agent’s response in real time. To run the UI:
#
# 1- Open a terminal and start the Ollama server: `ollama serve`.
#
# 2- In a second terminal, navigate to the frontend folder and install dependencies:`npm install`.
#
# 3- In the same terminal, start the FastAPI back‑end: `uvicorn app:app --reload --port 8000`
#
# 4- Open a third terminal, stay in the frontend folder, and start the React dev server: `npm run dev`
#
# 5- Visit `http://localhost:5173/` in your browser.
#

# %% [markdown]
#
# ## 🎉 Congratulations!
#
# * You have built a **web‑enabled agent**: tool calling → JSON schema → LangChain ReAct → web search → simple UI.
# * Try adding more tools, such as news or finance APIs.
# * Experiment with multiple tools, different models, and measure accuracy vs. hallucination.
#
#
# 👏 **Great job!** Take a moment to celebrate. The techniques you implemented here power many production agents and chatbots.
