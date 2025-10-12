# Project 1: Build an LLM Playground

Welcome! In this project, you’ll learn foundations of large language models (LLMs). We’ll keep the code minimal and the explanations high‑level so that anyone who can run a Python cell can follow along.  

We'll be using Google Colab for this project. Colab is a free, browser-based platform that lets you run Python code and machine learning models without installing anything on your local computer. Click the button below to open this notebook directly in Google Colab and get started!


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bytebyteai/ai-eng-projects/blob/main/project_1/lm_playground.ipynb)

---
## Learning Objectives  
* **Tokenization** and how raw text is tokenized into a sequene of discrete tokens
* Inspect **GPT2** and **Transformer architecture**
* Loading pre-trained LLMs using **Hugging Face**
* **Decoding strategies** to generate text from LLMs
* Completion versus **intrusction fine-tuned** LLMs


Let's get started!


```python
import torch, transformers, tiktoken
print("torch", torch.__version__, "| transformers", transformers.__version__)
```

    torch 2.8.0 | transformers 4.57.0


# 1 - Tokenization

A neural network can’t digest raw text. They need **numbers**. Tokenization is the process of converting text into IDs. In this section, you'll learn how tokenization is implemented in practice.

Tokenization methods generally fall into three categories:
1. Word-level
2. Character-level
3. Subword-level

### 1.1 - Word‑level tokenization

Split text on whitespace and store each **word** as a token.


```python
# 1. Tiny corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Tokenization converts text to numbers",
    "Large language models predict the next token"
]

# 2. Build the vocabulary
PAD, UNK = "[PAD]", "[UNK]"
vocab = []
word2id = {}
id2word = {}


# flatten all words, lowercased
tokens = []
for sentence in corpus:
    tokens.extend(sentence.lower().split())

# unique tokens
unique_tokens = sorted(set(tokens))

# prepend special tokens
vocab = [PAD, UNK] + unique_tokens

# make mapping dicts
word2id = {word: idx for idx, word in enumerate(vocab)}
id2word = {idx: word for word, idx in word2id.items()}


print(f"Vocabulary size: {len(vocab)} words")
print("First 15 vocab entries:", vocab[:15])

# 3. Encode / decode
def encode(text):
    tokens = text.lower().split()
    ids = [word2id.get(tok, word2id[UNK]) for tok in tokens]
    return ids

def decode(ids):
    words = [id2word.get(i, UNK) for i in ids]
    return " ".join(words)

# 4. Demo
sample = "The brown unicorn jumps"
ids = encode(sample)
recovered = decode(ids)

print("\nInput text :", sample)
print("Token IDs  :", ids)
print("Decoded    :", recovered)
```

    Vocabulary size: 21 words
    First 15 vocab entries: ['[PAD]', '[UNK]', 'brown', 'converts', 'dog', 'fox', 'jumps', 'language', 'large', 'lazy', 'models', 'next', 'numbers', 'over', 'predict']
    
    Input text : The brown unicorn jumps
    Token IDs  : [17, 2, 1, 6]
    Decoded    : the brown [UNK] jumps


Word-level tokenization has two major limitations:
1. Large vocabulary size
2. Out-of-vocabulary (OOV) issue

### 1.2 - Character‑level tokenization

Every single character (including spaces and emojis) gets its own ID. This guarantees zero out‑of‑vocabulary issues but very long sequences.


```python
# 1. Build a fixed vocabulary # a–z + A–Z + padding + unkwown
import string

PAD, UNK = "[PAD]", "[UNK]"

# all lowercase + uppercase letters
letters = list(string.ascii_lowercase + string.ascii_uppercase)
# print(letters)

vocab = [PAD, UNK] + letters
char2id = { ch: idx for idx, ch in enumerate(vocab)}
id2char = { idx: ch for ch, idx in char2id.items()}

# print(char2id)
# print(id2char)

# print(f"Vocabulary size: {len(vocab)} (52 letters + 2 specials)")

# # 2. Encode / decode
def encode(text):
      # turn each character into its ID (or UNK if not found)
    ids = [char2id.get(ch, char2id[UNK]) for ch in text]
    return ids
print (encode("Hello?"))

def decode(ids):
    # turn IDs back into characters
    chars = [id2char.get(i, UNK) for i in ids]
    return "".join(chars)

print (decode([35, 6, 13, 13, 16, 1]))

# # 3. Demo
# sample = "Hello"
# ids = encode(sample)
# recovered = decode(ids)

# print("\nInput text :", sample)
# print("Token IDs  :", ids)
# print("Decoded    :", recovered)

```

    [35, 6, 13, 13, 16, 1]
    Hello[UNK]


### 1.3 - Subword‑level tokenization

Sub-word methods such as `Byte-Pair Encoding (BPE)`, `WordPiece`, and `SentencePiece` **learn** the most common character and gorup them into new tokens. For example, the word `unbelievable` might turn into three tokens: `["un", "believ", "able"]`. This approach strikes a balance between word-level and character-level methods and fix their limitations.

For example, `BPE` algorithm forms the vocabulary using the following steps:
1. **Start with bytes** → every character is its own token.  
2. **Count all adjacent pairs** in a huge corpus.  
3. **Merge the most frequent pair** into a new token.  
   *Repeat steps 2-3* until you hit the target vocab size (e.g., 50 k).

Let's see `BPE` in practice.


```python
# 1. Load a pretrained BPE tokenizer (GPT-2 uses BPE).
# Refer to  https://huggingface.co/docs/transformers/en/fast_tokenizers

from transformers import AutoTokenizer

bpe_tok = AutoTokenizer.from_pretrained("gpt2")
if bpe_tok.pad_token is None:
    bpe_tok.add_special_tokens({"pad_token": "[PAD]"})


# 2. Encode / decode
def encode(text):
    # return a flat list of token IDs (no batch dimension)
    # we won't pad here since it's a simple demo
    return bpe_tok.encode(text, add_special_tokens=False)

ids = encode("The strawberry mojito was unbelievably delightful")
print (encode("The strawberry mojito was unbelievably delightful"))

print(bpe_tok.convert_ids_to_tokens(ids))

def decode(ids):
    """
    YOUR CODE HERE
    """

# # 3. Demo
# sample = "Unbelievable tokenization powers! 🚀"
# ids = encode(sample)
# recovered = decode(ids)

# print("\nInput text :", sample)
# print("Token IDs  :", ids)
# print("Tokens     :", bpe_tok.convert_ids_to_tokens(ids))
# print("Decoded    :", recovered)

```

    [464, 41236, 6941, 73, 10094, 373, 48943, 32327]
    ['2', '#', '"', "'"]


### 1.4 - TikToken

`tiktoken` is a production-ready library which offers high‑speed tokenization used by OpenAI models.  
Let's compare the older **gpt2** encoding with the newer **cl100k_base** used in GPT‑4.


```python
# Use gpt2 and cl100k_base to encode and decode the following text
# Refer to https://github.com/openai/tiktoken
import tiktoken

sentence = "The 🌟 star-player scored 40 points!"

"""
YOUR CODE HERE
"""
```

Experiment: try new sentences, emojis, code snippets, or other languages. If you are interested, try implementing the BPE algorithm yourself.

### 1.5 - Key Takeaways

* **Word‑level**: simple but brittle (OOV problems).  
* **Character‑level**: robust but produces long sequences.  
* **BPE / Byte‑Level BPE**: middle ground used by most LLMs.  
* **tiktoken**: shows how production models tokenize with pre‑trained sub‑word vocabularies.

# 2. What is a Language Model?

At its core, a **language model (LM)** is just a *very large* mathematical function built from many neural-network layers.  
Given a sequence of tokens `[t₁, t₂, …, tₙ]`, it learns to output a probability for the next token `tₙ₊₁`.


Each layer applies a simple operation (matrix multiplication, attention, etc.). Stacking hundreds of these layers lets the model capture patterns and statistical relations from text. The final output is a vector of scores that says, “how likely is each possible token to come next?”

> Think of the whole network as **one gigantic equation** whose parameters were tuned during training to minimize prediction error.



### 2.1 - A Single `Linear` Layer

Before we explore Transformer, let’s start tiny:

* A **Linear layer** performs `y = Wx + b`  
  * `x` – input vector  
  * `W` – weight matrix (learned)  
  * `b` – bias vector (learned)

Although this looks basic, chaining thousands of such linear transforms (with nonlinearities in between) gives neural nets their expressive power.



```python
import torch.nn as nn
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        """
        YOUR CODE HERE
        """

    def forward(self, x):
        """
        YOUR CODE HERE
        """
```


```python
import torch.nn as nn, torch

lin = nn.Linear(3, 2)
x = torch.tensor([1.0, -1.0, 0.5])
print("Input :", x)
print("Weights:", lin.weight)
print("Bias   :", lin.bias)
print("Output :", lin(x))

```

### 2.2 - A `Transformer` Layer

Most LLMs are a **stack of identical Transformer blocks**. Each block fuses two main components:

| Step | What it does | Where it lives in code |
|------|--------------|------------------------|
| **Multi-Head Self-Attention** | Every token looks at every other token and decides *what matters*. | `block.attn` |
| **Feed-Forward Network (MLP)** | Re-mixes information token-by-token. | `block.mlp` |

Below, we load the smallest public GPT-2 (124 M parameters), grab its *first* block, and inspect the pieces.



```python
import torch
from transformers import GPT2LMHeadModel

# Load the 124 M-parameter GPT-2 and inspect its layers (12 layers)
"""
YOUR CODE HERE
"""
```


```python
# Run a tiny forward pass through the first block
seq_len = 8
dummy_tokens = torch.randint(0, gpt2.config.vocab_size, (1, seq_len))
with torch.no_grad():
    # Embed tokens + positions the same way GPT-2 does
    # Forward through one layer
    """
    YOUR CODE HERE
    """

print("\nOutput shape :", out.shape) # (batch, seq_len, hidden_size)
```

### 2.3 - Inside GPT-2

GPT-2 is just many of those modules arranged in a repeating *block*. Let's print the modules inside the Transformer.


```python
# Print the name and modules inside gpt2
"""
YOUR CODE HERE
"""
```

As you can see, the Transformer holds various modules, arranged from a list of blocks (`h`). The following table summarizes these modules:

| Step | What it does | Why it matters |
|------|--------------|----------------|
| **Token → Embedding** | Converts IDs to vectors | Gives the model a numeric “handle” on words |
| **Positional Encoding** | Adds “where am I?” info | Order matters in language |
| **Multi-Head Self-Attention** | Each token asks “which other tokens should I look at?” | Lets the model relate words across a sentence |
| **Feed-Forward Network** | Two stacked Linear layers with a non-linearity | Mixes information and adds depth |
| **LayerNorm & Residual** | Stabilize training and help gradients flow | Keeps very deep networks trainable |


### 2.4 LLM's output

Passing a token sequence through an **LLM** yields a tensor of **logits** with shape  
`(batch_size, seq_len, vocab_size)`.  
Applying `softmax` on the last dimension turns those logits into probabilities.

The cell below feeds an 8-token dummy sequence, prints the logits shape, and shows the five most likely next tokens for the final position.



```python
import torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load gpt2 model and tokenizer
"""
YOUR CODE HERE
"""

# Tokenize input text
text = "Hello my name"
"""
YOUR CODE HERE
"""

# Get logits by passing the ids to the gpt2 model.
"""
YOUR CODE HERE
"""

print("Logits shape :", logits.shape)

# Predict next token
"""
YOUR CODE HERE
"""

print("\nTop-5 predictions for the next token:")
"""
YOUR CODE HERE
"""

```

### 2.5 - Key Takeaway

A language model is nothing mystical: it’s a *huge composition* of small, understandable layers trained to predict the next token in a sequence of tokens.

# 3 - Generation
Once an LLM is trained to predict the probabilities, we can generate text from the model. This process is called decoding or sampling.

At each step, the LLM outputs a **probability distribution** over the next token. It is the job of the decoding algorithm to pick the next token, and move on to the next token. There are different decoding algorithms and hyper-parameters to control the generaiton:
* **Greedy** → pick the single highest‑probability token each step (safe but repetitive).  
* **Top‑k / Nucleus (top‑p)** → sample from a subset of likely tokens (adds variety).
* **beam** -> applies beam search to pick tokens
* **Temperature** → a *creativity* knob. Higher values flatten the probability distribution.

### 3.1 - Greedy decoding


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
MODELS = {
    "gpt2": "gpt2",
}
tokenizers, models = {}, {}
# Load models and tokenizers
"""
YOUR CODE HERE
"""

def generate(model_key, prompt, strategy="greedy", max_new_tokens=100):
    tok, mdl = tokenizers[model_key], models[model_key]
    # Return the generations based on the provided strategy: greedy, top_k, top_p
    """
    YOUR CODE HERE
    """

```


```python
tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n== GPT-2 | Greedy ==")
    print(generate("gpt2", prompt, "greedy", 80))

```


Naively picking the single best token every time has the following issues in practice:

* **Loop**: “The cat is is is…”  
* **Miss long-term payoff**: the highest-probability word *now* might paint you into a boring corner later.

### 3.2 - Top-k or top-p sampling


```python

tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n== GPT-2 | Top-p ==")
    print(generate("gpt2", prompt, "top-p", 40))

```

### 3.3 - Try It Yourself

1. Scroll to the list called `tests`.
2. Swap in your own prompts or tweak the decoding strategy.  
3. Re‑run the cell and compare the vibes.

> **Tip:** Try the same prompt with `greedy` vs. `top_p` (0.9) and see how the tone changes. Notice especially how small temperature tweaks can soften or sharpen the prose.

* `strategy`: `"greedy"`, `"beam"`, `"top_k"`, `"top_p"`  
* `temperature`: `0.2 – 2.0`  
* `k` or `p` thresholds



# 4 - Completion vs. Instruction-tuned LLMs

We have seen that we can use GPT2 model to pass an input text and generate a new text. However, this model only continues the provided text. It is not engaging in a dialouge-like conversation and cannot be helpful by answering instructions. On the other hand, **instruction-tuned LLMs** like `Qwen-Chat` go through an extra training stage called **post-training** after the base “completion” model is finished. Because of post-training step, an instruction-tuned LLM will:

* **Read the entire prompt as a request,** not just as text to mimic.  
* **Stay in dialogue mode**. Answer questions, follow steps, ask clarifying queries.  
* **Refuse or safe-complete** when instructions are unsafe or disallowed.  
* **Adopt a consistent persona** (e.g., “Assistant”) rather than drifting into story continuation.


### 4.1 - Qwen1.5-8B vs. GPT2

In the code below we’ll feed the same prompt to:

* **GPT-2 (completion-only)** – it will simply keep writing in the same style.  
* **Qwen-Chat (instruction-tuned)** – it will obey the instruction and respond directly.

Comparing the two outputs makes the difference easy to see.


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
MODELS = {
    "gpt2": "gpt2",
    "qwen": "Qwen/Qwen1.5-1.8B-Chat"
}
tokenizers, models = {}, {}
# Load models and tokenizers
"""
YOUR CODE HERE
"""

```


We downloaded two tiny checkpoints: `GPT‑2` (124 M parameters) and `Qwen‑1.5‑Chat` (1.8 B). If the cell took a while, that was mostly network time. Models are stored locally after the first run.

Let's now generate text and compare two models.



```python

tests=[("Once upon a time","greedy"),("What is 2+2?","top_k"),("Suggest a party theme.","top_p")]
for prompt,strategy in tests:
    for key in ["gpt2","qwen"]:
        print(f"\n== {key.upper()} | {strategy} ==")
        print(generate(key,prompt,strategy,80))

```

# 5. (Optional) A Small LLM Playground

### 5.1 ‑ Interactive Playground

Enter a prompt, pick a model and decoding strategy, adjust the temperature, and press **Generate** to watch the model respond.



```python
import ipywidgets as widgets
from IPython.display import display, Markdown

# Make sure models and tokenizers are loaded
try:
    tokenizers
    models
except NameError:
    raise RuntimeError("Please run the earlier setup cells that load the models before using the playground.")

def generate_playground(model_key, prompt, strategy="greedy", temperature=1.0, max_new_tokens=100):
    # Generation code
    """
    YOUR CODE HERE
    """

# Your code to build boxes, dropdowns, and other elements in the UI using widgets and creating the UI using widgets.vbox and display.
# Refer to https://ipywidgets.readthedocs.io/en/stable/
"""
YOUR CODE HERE
"""

```


## 🎉 Congratulations!

You’ve just learned, explored, and inspected a real **LLM**. In one project you:
* Learned how **tokenization** works in practice
* Used `tiktoken` library to load and experiment with most advanced tokenizers.
* Explored LLM architecture and inspected GPT2 blocks and layers
* Learned decoding strategies and used `top-p` to generate text from GPT2
* Loaded a powerful chat model, `Qwen1.5-8B` and generated text
* Built an LLM playground


👏 **Great job!** Take a moment to celebrate. You now have a working mental model of how LLMs work. The skills you used here power most LLMs you see everywhere.



