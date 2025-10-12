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
     # convert a list of IDs back to text
    return bpe_tok.decode(ids, skip_special_tokens=True)

print (decode([464, 41236, 6941, 73, 10094, 373, 48943, 32327]))

# # 3. Demo
sample = "Unbelievable tokenization powers! 🚀"
ids = encode(sample)
recovered = decode(ids)

print("\nInput text :", sample)
print("Token IDs  :", ids)
print("Tokens     :", bpe_tok.convert_ids_to_tokens(ids))
print("Decoded    :", recovered)

```

    [464, 41236, 6941, 73, 10094, 373, 48943, 32327]
    ['The', 'Ġstrawberry', 'Ġmo', 'j', 'ito', 'Ġwas', 'Ġunbelievably', 'Ġdelightful']
    The strawberry mojito was unbelievably delightful
    
    Input text : Unbelievable tokenization powers! 🚀
    Token IDs  : [3118, 6667, 11203, 540, 11241, 1634, 5635, 0, 12520, 248, 222]
    Tokens     : ['Un', 'bel', 'iev', 'able', 'Ġtoken', 'ization', 'Ġpowers', '!', 'ĠðŁ', 'ļ', 'Ģ']
    Decoded    : Unbelievable tokenization powers! 🚀


### 1.4 - TikToken

`tiktoken` is a production-ready library which offers high‑speed tokenization used by OpenAI models.  
Let's compare the older **gpt2** encoding with the newer **cl100k_base** used in GPT‑4.


```python
# Use gpt2 and cl100k_base to encode and decode the following text
# Refer to https://github.com/openai/tiktoken
import tiktoken

sentence = "The 🌟 star-player scored 40 points!"

enc_gpt2 = tiktoken.get_encoding("gpt2")
enc_cl100k = tiktoken.get_encoding("cl100k_base")

# Encode (convert text → token IDs)
ids_gpt2 = enc_gpt2.encode(sentence)
ids_cl100k = enc_cl100k.encode(sentence)

# Decode (convert token IDs → text)
decoded_gpt2 = enc_gpt2.decode(ids_gpt2)
decoded_cl100k = enc_cl100k.decode(ids_cl100k)

# Print results
print("Sentence:", sentence, "\n")

print("GPT-2 encoding:")
print("IDs:", ids_gpt2)
print("Tokens:", [enc_gpt2.decode([i]) for i in ids_gpt2])
print("Decoded:", decoded_gpt2, "\n")

print("cl100k_base (GPT-4) encoding:")
print("IDs:", ids_cl100k)
print("Tokens:", [enc_cl100k.decode([i]) for i in ids_cl100k])
print("Decoded:", decoded_cl100k)
```

    Sentence: The 🌟 star-player scored 40 points! 
    
    GPT-2 encoding:
    IDs: [464, 12520, 234, 253, 3491, 12, 7829, 7781, 2319, 2173, 0]
    Tokens: ['The', ' �', '�', '�', ' star', '-', 'player', ' scored', ' 40', ' points', '!']
    Decoded: The 🌟 star-player scored 40 points! 
    
    cl100k_base (GPT-4) encoding:
    IDs: [791, 11410, 234, 253, 6917, 43467, 16957, 220, 1272, 3585, 0]
    Tokens: ['The', ' �', '�', '�', ' star', '-player', ' scored', ' ', '40', ' points', '!']
    Decoded: The 🌟 star-player scored 40 points!


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
        # initialize weights and bias as parameters so they can be learned
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # perform y = W * x + b
        return self.W @ x + self.b
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

    Input : tensor([ 1.0000, -1.0000,  0.5000])
    Weights: Parameter containing:
    tensor([[-0.4009,  0.1071, -0.3196],
            [ 0.2048, -0.1436, -0.0861]], requires_grad=True)
    Bias   : Parameter containing:
    tensor([ 0.4657, -0.2889], requires_grad=True)
    Output : tensor([-0.2022,  0.0165], grad_fn=<ViewBackward0>)


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
# Load the 124M GPT-2 (12 blocks, 12 heads, 768 hidden)
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
print("layers:", gpt2.config.n_layer,
      "| heads:", gpt2.config.n_head,
      "| hidden:", gpt2.config.n_embd)

# This is GPT-2’s architecture summary:

# Term	Meaning
# layers = 12	GPT-2 has 12 Transformer blocks stacked one after another. Each block processes the same sequence, adding more depth and abstraction each time.
# heads = 12	Each block’s self-attention is split into 12 “heads,” so the model can focus on 12 different relationship patterns between tokens at once.
# hidden = 768	Every token is represented as a 768-dimensional vector inside the network (that’s the size of the embedding and of each block’s hidden layer).

# So, for every word or token, GPT-2 stores a vector of length 768 and transforms it 12 times, each time using 12 attention heads.

# Grab the Transformer stack and the FIRST block
stack = gpt2.transformer              # embeddings + blocks + final ln_f
block0 = gpt2.transformer.h[0]        # first Transformer block
print(block0)                         # shows attn + mlp inside the block

# 🔁 How a GPT-2 block works

# Normalize the input → ln_1

# Self-attention → tokens communicate and exchange info

# Add residual connection (input + attention output)

# Normalize again → ln_2

# Feed-forward network (mlp) → transforms each token individually

# Add another residual

# That’s one complete Transformer block.
# GPT-2 stacks 12 of these in a row → that’s what gives it depth and “intelligence.”

# 🔍 Quick analogy

# Think of each block as:

# 👁️ The attention part decides what other words each token should “look at.”

# 🧠 The MLP part reasons about what it just saw and updates its internal representation.

# ⛓️ Repeating 12 times lets the model build deeper understanding of the whole sentence.
```


    model.safetensors:   0%|          | 0.00/548M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]


    layers: 12 | heads: 12 | hidden: 768
    GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D(nf=2304, nx=768)
        (c_proj): Conv1D(nf=768, nx=768)
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D(nf=3072, nx=768)
        (c_proj): Conv1D(nf=768, nx=3072)
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )



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
for name, module in gpt2.named_modules():
    print(name, ":", module.__class__.__name__)
```

     : GPT2LMHeadModel
    transformer : GPT2Model
    transformer.wte : Embedding
    transformer.wpe : Embedding
    transformer.drop : Dropout
    transformer.h : ModuleList
    transformer.h.0 : GPT2Block
    transformer.h.0.ln_1 : LayerNorm
    transformer.h.0.attn : GPT2Attention
    transformer.h.0.attn.c_attn : Conv1D
    transformer.h.0.attn.c_proj : Conv1D
    transformer.h.0.attn.attn_dropout : Dropout
    transformer.h.0.attn.resid_dropout : Dropout
    transformer.h.0.ln_2 : LayerNorm
    transformer.h.0.mlp : GPT2MLP
    transformer.h.0.mlp.c_fc : Conv1D
    transformer.h.0.mlp.c_proj : Conv1D
    transformer.h.0.mlp.act : NewGELUActivation
    transformer.h.0.mlp.dropout : Dropout
    transformer.h.1 : GPT2Block
    transformer.h.1.ln_1 : LayerNorm
    transformer.h.1.attn : GPT2Attention
    transformer.h.1.attn.c_attn : Conv1D
    transformer.h.1.attn.c_proj : Conv1D
    transformer.h.1.attn.attn_dropout : Dropout
    transformer.h.1.attn.resid_dropout : Dropout
    transformer.h.1.ln_2 : LayerNorm
    transformer.h.1.mlp : GPT2MLP
    transformer.h.1.mlp.c_fc : Conv1D
    transformer.h.1.mlp.c_proj : Conv1D
    transformer.h.1.mlp.act : NewGELUActivation
    transformer.h.1.mlp.dropout : Dropout
    transformer.h.2 : GPT2Block
    transformer.h.2.ln_1 : LayerNorm
    transformer.h.2.attn : GPT2Attention
    transformer.h.2.attn.c_attn : Conv1D
    transformer.h.2.attn.c_proj : Conv1D
    transformer.h.2.attn.attn_dropout : Dropout
    transformer.h.2.attn.resid_dropout : Dropout
    transformer.h.2.ln_2 : LayerNorm
    transformer.h.2.mlp : GPT2MLP
    transformer.h.2.mlp.c_fc : Conv1D
    transformer.h.2.mlp.c_proj : Conv1D
    transformer.h.2.mlp.act : NewGELUActivation
    transformer.h.2.mlp.dropout : Dropout
    transformer.h.3 : GPT2Block
    transformer.h.3.ln_1 : LayerNorm
    transformer.h.3.attn : GPT2Attention
    transformer.h.3.attn.c_attn : Conv1D
    transformer.h.3.attn.c_proj : Conv1D
    transformer.h.3.attn.attn_dropout : Dropout
    transformer.h.3.attn.resid_dropout : Dropout
    transformer.h.3.ln_2 : LayerNorm
    transformer.h.3.mlp : GPT2MLP
    transformer.h.3.mlp.c_fc : Conv1D
    transformer.h.3.mlp.c_proj : Conv1D
    transformer.h.3.mlp.act : NewGELUActivation
    transformer.h.3.mlp.dropout : Dropout
    transformer.h.4 : GPT2Block
    transformer.h.4.ln_1 : LayerNorm
    transformer.h.4.attn : GPT2Attention
    transformer.h.4.attn.c_attn : Conv1D
    transformer.h.4.attn.c_proj : Conv1D
    transformer.h.4.attn.attn_dropout : Dropout
    transformer.h.4.attn.resid_dropout : Dropout
    transformer.h.4.ln_2 : LayerNorm
    transformer.h.4.mlp : GPT2MLP
    transformer.h.4.mlp.c_fc : Conv1D
    transformer.h.4.mlp.c_proj : Conv1D
    transformer.h.4.mlp.act : NewGELUActivation
    transformer.h.4.mlp.dropout : Dropout
    transformer.h.5 : GPT2Block
    transformer.h.5.ln_1 : LayerNorm
    transformer.h.5.attn : GPT2Attention
    transformer.h.5.attn.c_attn : Conv1D
    transformer.h.5.attn.c_proj : Conv1D
    transformer.h.5.attn.attn_dropout : Dropout
    transformer.h.5.attn.resid_dropout : Dropout
    transformer.h.5.ln_2 : LayerNorm
    transformer.h.5.mlp : GPT2MLP
    transformer.h.5.mlp.c_fc : Conv1D
    transformer.h.5.mlp.c_proj : Conv1D
    transformer.h.5.mlp.act : NewGELUActivation
    transformer.h.5.mlp.dropout : Dropout
    transformer.h.6 : GPT2Block
    transformer.h.6.ln_1 : LayerNorm
    transformer.h.6.attn : GPT2Attention
    transformer.h.6.attn.c_attn : Conv1D
    transformer.h.6.attn.c_proj : Conv1D
    transformer.h.6.attn.attn_dropout : Dropout
    transformer.h.6.attn.resid_dropout : Dropout
    transformer.h.6.ln_2 : LayerNorm
    transformer.h.6.mlp : GPT2MLP
    transformer.h.6.mlp.c_fc : Conv1D
    transformer.h.6.mlp.c_proj : Conv1D
    transformer.h.6.mlp.act : NewGELUActivation
    transformer.h.6.mlp.dropout : Dropout
    transformer.h.7 : GPT2Block
    transformer.h.7.ln_1 : LayerNorm
    transformer.h.7.attn : GPT2Attention
    transformer.h.7.attn.c_attn : Conv1D
    transformer.h.7.attn.c_proj : Conv1D
    transformer.h.7.attn.attn_dropout : Dropout
    transformer.h.7.attn.resid_dropout : Dropout
    transformer.h.7.ln_2 : LayerNorm
    transformer.h.7.mlp : GPT2MLP
    transformer.h.7.mlp.c_fc : Conv1D
    transformer.h.7.mlp.c_proj : Conv1D
    transformer.h.7.mlp.act : NewGELUActivation
    transformer.h.7.mlp.dropout : Dropout
    transformer.h.8 : GPT2Block
    transformer.h.8.ln_1 : LayerNorm
    transformer.h.8.attn : GPT2Attention
    transformer.h.8.attn.c_attn : Conv1D
    transformer.h.8.attn.c_proj : Conv1D
    transformer.h.8.attn.attn_dropout : Dropout
    transformer.h.8.attn.resid_dropout : Dropout
    transformer.h.8.ln_2 : LayerNorm
    transformer.h.8.mlp : GPT2MLP
    transformer.h.8.mlp.c_fc : Conv1D
    transformer.h.8.mlp.c_proj : Conv1D
    transformer.h.8.mlp.act : NewGELUActivation
    transformer.h.8.mlp.dropout : Dropout
    transformer.h.9 : GPT2Block
    transformer.h.9.ln_1 : LayerNorm
    transformer.h.9.attn : GPT2Attention
    transformer.h.9.attn.c_attn : Conv1D
    transformer.h.9.attn.c_proj : Conv1D
    transformer.h.9.attn.attn_dropout : Dropout
    transformer.h.9.attn.resid_dropout : Dropout
    transformer.h.9.ln_2 : LayerNorm
    transformer.h.9.mlp : GPT2MLP
    transformer.h.9.mlp.c_fc : Conv1D
    transformer.h.9.mlp.c_proj : Conv1D
    transformer.h.9.mlp.act : NewGELUActivation
    transformer.h.9.mlp.dropout : Dropout
    transformer.h.10 : GPT2Block
    transformer.h.10.ln_1 : LayerNorm
    transformer.h.10.attn : GPT2Attention
    transformer.h.10.attn.c_attn : Conv1D
    transformer.h.10.attn.c_proj : Conv1D
    transformer.h.10.attn.attn_dropout : Dropout
    transformer.h.10.attn.resid_dropout : Dropout
    transformer.h.10.ln_2 : LayerNorm
    transformer.h.10.mlp : GPT2MLP
    transformer.h.10.mlp.c_fc : Conv1D
    transformer.h.10.mlp.c_proj : Conv1D
    transformer.h.10.mlp.act : NewGELUActivation
    transformer.h.10.mlp.dropout : Dropout
    transformer.h.11 : GPT2Block
    transformer.h.11.ln_1 : LayerNorm
    transformer.h.11.attn : GPT2Attention
    transformer.h.11.attn.c_attn : Conv1D
    transformer.h.11.attn.c_proj : Conv1D
    transformer.h.11.attn.attn_dropout : Dropout
    transformer.h.11.attn.resid_dropout : Dropout
    transformer.h.11.ln_2 : LayerNorm
    transformer.h.11.mlp : GPT2MLP
    transformer.h.11.mlp.c_fc : Conv1D
    transformer.h.11.mlp.c_proj : Conv1D
    transformer.h.11.mlp.act : NewGELUActivation
    transformer.h.11.mlp.dropout : Dropout
    transformer.ln_f : LayerNorm
    lm_head : Linear


As you can see, the Transformer holds various modules, arranged from a list of blocks (`h`). The following table summarizes these modules:

| Step | What it does | Why it matters |
|------|--------------|----------------|
| **Token → Embedding** | Converts IDs to vectors | Gives the model a numeric “handle” on words |
| **Positional Encoding** | Adds “where am I?” info | Order matters in language |
| **Multi-Head Self-Attention** | Each token asks “which other tokens should I look at?” | Lets the model relate words across a sentence |
| **Feed-Forward Network** | Two stacked Linear layers with a non-linearity | Mixes information and adds depth |
| **LayerNorm & Residual** | Stabilize training and help gradients flow | Keeps very deep networks trainable |


🧩 What this shows

transformer.wte → word (token) embedding layer

transformer.wpe → positional embedding layer

transformer.h.[i] → 12 identical Transformer blocks

each has attn (attention), mlp (feed-forward), and layernorms

transformer.ln_f → final normalization

lm_head → linear layer that maps hidden states back to vocab logits for predicting the next token

So, GPT-2 is literally:

[ Embeddings ]
+ [ 12 Transformer Blocks ]
+ [ Output head ]

### 2.4 LLM's output

Passing a token sequence through an **LLM** yields a tensor of **logits** with shape  
`(batch_size, seq_len, vocab_size)`.  
Applying `softmax` on the last dimension turns those logits into probabilities.

The cell below feeds an 8-token dummy sequence, prints the logits shape, and shows the five most likely next tokens for the final position.



```python
import torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load gpt2 model and tokenizer
model_name = "gpt2"
gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

# Tokenize input text
text = "Hello my name"
text = "Hello my name"
inputs = tokenizer(text, return_tensors="pt")   # gives input_ids + attention_mask
input_ids = inputs["input_ids"]

# Get logits by passing the ids to the gpt2 model.
with torch.no_grad():
    outputs = gpt2(input_ids)
logits = outputs.logits                        # shape: (batch, seq_len, vocab_size)

print("Logits shape :", logits.shape)

# Predict next token
# Take the last position's logits ([-1] = last token)
next_token_logits = logits[0, -1, :]

# Convert logits → probabilities using softmax
probs = F.softmax(next_token_logits, dim=-1)

print("\nTop-5 predictions for the next token:")
top_k = 5
top_probs, top_ids = torch.topk(probs, top_k)

print("\nTop-5 predictions for the next token:")
for i in range(top_k):
    token = tokenizer.decode(top_ids[i])
    print(f"{i+1}. {token!r}  (prob={top_probs[i]:.4f})")


# 🔢 Sequence of Events: How GPT-2 Generates Next-Token Predictions

# Start with text — a short phrase or sentence.

# Tokenize — convert the text into numeric token IDs that the model understands.

# Feed tokens into the model — pass the token IDs through GPT-2.

# Model outputs logits — raw, unnormalized scores for every token in the vocabulary, for every position in the sequence. 

# Focus on the last position’s logits — these represent the model’s prediction for the next token. That gives you one vector with 50k numbers — one score for every possible next token in GPT-2’s vocabulary.

# Then we apply softmax to turn those scores into actual probabilities:

# Apply softmax — transform logits into probabilities that sum to 1.

# Sort or select top-k — pick the most likely next tokens based on those probabilities.

# Decode — turn the predicted token IDs back into readable text if needed.

```

    '(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: b45c4307-3258-4c09-b031-07e48acef3c1)')' thrown while requesting HEAD https://huggingface.co/gpt2/resolve/main/config.json
    Retrying in 1s [Retry 1/5].


    Logits shape : torch.Size([1, 3, 50257])
    
    Top-5 predictions for the next token:
    
    Top-5 predictions for the next token:
    1. ' is'  (prob=0.7774)
    2. ','  (prob=0.0373)
    3. "'s"  (prob=0.0332)
    4. ' was'  (prob=0.0127)
    5. ' and'  (prob=0.0076)


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
for key, name in MODELS.items():
    tokenizers[key] = AutoTokenizer.from_pretrained(name)
    models[key] = AutoModelForCausalLM.from_pretrained(name)

def generate(model_key, prompt, strategy="greedy", max_new_tokens=100):
    tok, mdl = tokenizers[model_key], models[model_key]
    # Return the generations based on the provided strategy: greedy, top_k, top_p
    inputs = tok(prompt, return_tensors="pt")

    if strategy == "greedy":
        # Pick the single most probable token each step
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # disables randomness
            temperature=1.0
        )

    elif strategy == "top_k":
        # Sample only from the top K likely tokens
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,             # only consider top 50 tokens
            temperature=0.8
        )

    elif strategy == "top_p":
        # Sample from smallest set of tokens whose cumulative prob ≥ p
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,            # nucleus sampling
            temperature=0.8
        )

    else:
        raise ValueError("Unknown strategy: choose from 'greedy', 'top_k', 'top_p'.")

    return tok.decode(outputs[0], skip_special_tokens=True)
    

```


```python
tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n== GPT-2 | Greedy ==")
    print(generate("gpt2", prompt, "greedy", 80))

```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    
    == GPT-2 | Greedy ==


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    Once upon a time, the world was a place of great beauty and great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and
    
    == GPT-2 | Greedy ==


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    What is 2+2?
    
    2+2 is the number of times you can use a spell to cast a spell.
    
    2+2 is the number of times you can use a spell to cast a spell.
    
    2+2 is the number of times you can use a spell to cast a spell.
    
    2+2 is the number of times you can use a spell to cast a spell.
    
    == GPT-2 | Greedy ==
    Suggest a party theme.
    
    The party theme is a simple, simple, and fun way to get your friends to join you.
    
    The party theme is a simple, simple, and fun way to get your friends to join you. The party theme is a simple, simple, and fun way to get your friends to join you. The party theme is a simple, simple, and fun way to get your friends



Naively picking the single best token every time has the following issues in practice:

* **Loop**: “The cat is is is…”  
* **Miss long-term payoff**: the highest-probability word *now* might paint you into a boring corner later.

### 3.2 - Top-k or top-p sampling


```python

tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n== GPT-2 | Top-p ==")
    print(generate("gpt2", prompt, "top_p", 40))

```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    
    == GPT-2 | Top-p ==


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    Once upon a time, he had been a member of the Council, but was no longer able to attend it, and was banished from the city.
    
    As he was about to leave, he heard a voice in
    
    == GPT-2 | Top-p ==


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    What is 2+2?
    
    When a player takes an action that would otherwise be considered a blocking action, they may take a penalty, or they may take an action on their turn that would otherwise be considered a blocking action
    
    == GPT-2 | Top-p ==
    Suggest a party theme.
    
    Get started with a simple app with a lot of customization.
    
    Let's start with the App
    
    A simple app with a few features is just going to take a little bit of


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
for key, name in MODELS.items():
    # Qwen chat models often need trust_remote_code=True for proper generation utils
    needs_trust = "Qwen" in name or "qwen" in name
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=needs_trust)
    mdl = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=needs_trust)

    # GPT-2 doesn’t have a pad token; set pad→eos to silence warnings
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token

    tokenizers[key] = tok
    models[key] = mdl

```


    tokenizer_config.json: 0.00B [00:00, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    config.json:   0%|          | 0.00/662 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/3.67G [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/206 [00:00<?, ?B/s]



We downloaded two tiny checkpoints: `GPT‑2` (124 M parameters) and `Qwen‑1.5‑Chat` (1.8 B). If the cell took a while, that was mostly network time. Models are stored locally after the first run.

Let's now generate text and compare two models.



```python

tests=[("Once upon a time","greedy"),("What is 2+2?","top_k"),("Suggest a party theme.","top_p")]
for prompt,strategy in tests:
    for key in ["gpt2","qwen"]:
        print(f"\n== {key.upper()} | {strategy} ==")
        print(generate(key,prompt,strategy,80))

```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    
    == GPT2 | greedy ==


    The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.


    Once upon a time, the world was a place of great beauty and great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and
    
    == QWEN | greedy ==


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    Once upon a time, there was a young girl named Lily who lived in a small village nestled among the rolling hills of her home. Lily had always been fascinated by nature and spent most of her days exploring the surrounding forests, streams, and meadows. She loved to sing songs about the birds that sang in the trees, the flowers that bloomed in the fields, and the animals that roamed free.
    One day
    
    == GPT2 | top_k ==
    What is 2+2?
    
    If there is no two people, or even an entire nation, in love, then it's obvious that there has been a lot of discord. No matter what you say, or whether you're a pro-gay marriage advocate or pro-choice, it's not surprising that people will question your views. So what does it mean for a man, a woman, as a human being to
    
    == QWEN | top_k ==


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    What is 2+2? The sum of 2 and 2 is 4. Therefore, the answer to the question "2+2" is 4. Is there anything else I can help you with?
    
    == GPT2 | top_p ==
    Suggest a party theme.
    
    A party theme should be one that uses the main character as the main character. This should look like:
    
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55
    
    == QWEN | top_p ==
    Suggest a party theme. "Enchanted Castle" Party Theme The Enchanted Castle theme is perfect for any magical-themed birthday party or event that wants to transport the guests to a world of fantasy and adventure. Here are some ideas for incorporating elements of an enchanted castle into your party:
    
    1. Decorations: Begin by decorating your party space with the main focus being on the castle itself. Use streamers, balloons,


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
    tok, mdl = tokenizers[model_key], models[model_key]

    # Some tokenizers (e.g., GPT-2) lack PAD; map PAD -> EOS once to avoid warnings
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token = tok.eos_token

    # Use chat template for instruction-tuned chat models when available (e.g., Qwen)
    if model_key.lower().startswith("qwen") or hasattr(tok, "apply_chat_template"):
        try:
            chat = [{"role": "user", "content": prompt}]
            formatted = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tok(formatted, return_tensors="pt")
        except Exception:
            inputs = tok(prompt, return_tensors="pt")
    else:
        inputs = tok(prompt, return_tensors="pt")

    gen_cfg = dict(
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tok.eos_token_id if getattr(tok, "eos_token_id", None) is not None else None,
    )

    # Decoding strategies
    if strategy == "greedy":
        gen_cfg.update(dict(do_sample=False))
    elif strategy == "top_k":
        gen_cfg.update(dict(do_sample=True, top_k=50, temperature=float(temperature)))
    elif strategy == "top_p":
        gen_cfg.update(dict(do_sample=True, top_p=0.9, temperature=float(temperature)))
    else:
        raise ValueError("strategy must be one of: 'greedy', 'top_k', 'top_p'")

    # Light repetition controls help with small models
    gen_cfg.update(dict(repetition_penalty=1.15, no_repeat_ngram_size=3))

    with torch.no_grad():
        out_ids = models[model_key].generate(**inputs, **gen_cfg)

    return tok.decode(out_ids[0], skip_special_tokens=True)

# Your code to build boxes, dropdowns, and other elements in the UI using widgets and creating the UI using widgets.vbox and display.
# Refer to https://ipywidgets.readthedocs.io/en/stable/
# ---------------- UI WIDGETS ----------------
model_dd = widgets.Dropdown(
    options=list(models.keys()),
    value=next(iter(models.keys())),
    description="Model:",
    layout=widgets.Layout(width="300px")
)

strategy_dd = widgets.Dropdown(
    options=[("Greedy", "greedy"), ("Top-k", "top_k"), ("Top-p (nucleus)", "top_p")],
    value="greedy",
    description="Decoding:",
    layout=widgets.Layout(width="300px")
)

temp_slider = widgets.FloatSlider(
    value=1.0, min=0.2, max=1.5, step=0.05,
    description="Temperature:",
    readout_format=".2f",
    continuous_update=False,
    layout=widgets.Layout(width="400px")
)

max_tokens_slider = widgets.IntSlider(
    value=80, min=1, max=256, step=1,
    description="Max new tokens:",
    continuous_update=False,
    layout=widgets.Layout(width="400px")
)

prompt_ta = widgets.Textarea(
    value="Suggest a party theme with decor, music, and snack ideas.",
    placeholder="Type your prompt here…",
    description="Prompt:",
    layout=widgets.Layout(width="100%", height="120px")
)

generate_btn = widgets.Button(
    description="Generate",
    button_style="primary",
    tooltip="Run the model",
    icon="play"
)

out = widgets.Output()


def on_generate_clicked(_):
    out.clear_output()
    with out:
        print("Generating…")
        try:
            text = generate_playground(
                model_key=model_dd.value,
                prompt=prompt_ta.value,
                strategy=strategy_dd.value,
                temperature=temp_slider.value,
                max_new_tokens=max_tokens_slider.value,
            )
            display(Markdown(f"### Output\n\n{text}"))
        except Exception as e:
            display(Markdown(f"**Error:** `{e}`"))

generate_btn.on_click(on_generate_clicked)

ui = widgets.VBox([
    widgets.HBox([model_dd, strategy_dd]),
    temp_slider,
    max_tokens_slider,
    prompt_ta,
    generate_btn,
    out
])

display(ui)


```


    VBox(children=(HBox(children=(Dropdown(description='Model:', layout=Layout(width='300px'), options=('gpt2', 'q…



## 🎉 Congratulations!

You’ve just learned, explored, and inspected a real **LLM**. In one project you:
* Learned how **tokenization** works in practice
* Used `tiktoken` library to load and experiment with most advanced tokenizers.
* Explored LLM architecture and inspected GPT2 blocks and layers
* Learned decoding strategies and used `top-p` to generate text from GPT2
* Loaded a powerful chat model, `Qwen1.5-8B` and generated text
* Built an LLM playground


👏 **Great job!** Take a moment to celebrate. You now have a working mental model of how LLMs work. The skills you used here power most LLMs you see everywhere.



