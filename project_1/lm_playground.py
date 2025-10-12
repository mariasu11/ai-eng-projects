# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (ai-eng-p311)
#     language: python
#     name: ai-eng-p311
# ---

# %% [markdown] id="fe523821"
#
# # Project 1: Build an LLM Playground
#
# Welcome! In this project, you’ll learn foundations of large language models (LLMs). We’ll keep the code minimal and the explanations high‑level so that anyone who can run a Python cell can follow along.  
#
# We'll be using Google Colab for this project. Colab is a free, browser-based platform that lets you run Python code and machine learning models without installing anything on your local computer. Click the button below to open this notebook directly in Google Colab and get started!
#

# %% [markdown] id="fdb8584e"
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bytebyteai/ai-eng-projects/blob/main/project_1/lm_playground.ipynb)

# %% [markdown] id="08e82492"
# ---
# ## Learning Objectives  
# * **Tokenization** and how raw text is tokenized into a sequene of discrete tokens
# * Inspect **GPT2** and **Transformer architecture**
# * Loading pre-trained LLMs using **Hugging Face**
# * **Decoding strategies** to generate text from LLMs
# * Completion versus **intrusction fine-tuned** LLMs
#
#
# Let's get started!

# %% colab={"base_uri": "https://localhost:8080/"} id="1235110e" outputId="b0de1e27-58b4-4749-f0ed-bc7045fdcae8"
import torch, transformers, tiktoken
print("torch", torch.__version__, "| transformers", transformers.__version__)

# %% [markdown] id="d4c1eb0b"
# # 1 - Tokenization
#
# A neural network can’t digest raw text. They need **numbers**. Tokenization is the process of converting text into IDs. In this section, you'll learn how tokenization is implemented in practice.
#
# Tokenization methods generally fall into three categories:
# 1. Word-level
# 2. Character-level
# 3. Subword-level

# %% [markdown] id="1d234dc0"
# ### 1.1 - Word‑level tokenization
#
# Split text on whitespace and store each **word** as a token.

# %% colab={"base_uri": "https://localhost:8080/"} id="d784a288" outputId="165c06bb-6fb6-4bbc-e748-82dd7f28f419"
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

# %% [markdown] id="0edab2c2"
# Word-level tokenization has two major limitations:
# 1. Large vocabulary size
# 2. Out-of-vocabulary (OOV) issue

# %% [markdown] id="a379bac7"
# ### 1.2 - Character‑level tokenization
#
# Every single character (including spaces and emojis) gets its own ID. This guarantees zero out‑of‑vocabulary issues but very long sequences.

# %% colab={"base_uri": "https://localhost:8080/"} id="4ac29144" outputId="4747118a-819f-4a7e-ffe7-bf39ad0b7216"
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


# %% [markdown] id="391275bd"
# ### 1.3 - Subword‑level tokenization
#
# Sub-word methods such as `Byte-Pair Encoding (BPE)`, `WordPiece`, and `SentencePiece` **learn** the most common character and gorup them into new tokens. For example, the word `unbelievable` might turn into three tokens: `["un", "believ", "able"]`. This approach strikes a balance between word-level and character-level methods and fix their limitations.
#
# For example, `BPE` algorithm forms the vocabulary using the following steps:
# 1. **Start with bytes** → every character is its own token.  
# 2. **Count all adjacent pairs** in a huge corpus.  
# 3. **Merge the most frequent pair** into a new token.  
#    *Repeat steps 2-3* until you hit the target vocab size (e.g., 50 k).
#
# Let's see `BPE` in practice.

# %% colab={"base_uri": "https://localhost:8080/", "height": 429, "referenced_widgets": ["aad14a73e2e14cbc81387b875efcf96f", "c0d4b3d62e5145bc985bdce2ac716f11", "b3c8ff37ccf7487d9d8eec960541e767", "fdb7dc6fff554b84a8d8291021c8b5a5", "0fdba089b729412db0a32206bb724e0f", "8a0bd05fba6b408caa56e2237bbb4a82", "e024652945f94fd2bc2d9a28760cb159", "d9722550d5fb42609da0b04985a55863", "d6097acce80d4aed855f6b9bbdf27f54", "ced878c887b448b1af8c9d8b45082572", "60acdab3e5104128a42264a7c8780316", "759aecff52b340ae8d5e7b878dda9a38", "485825bd7f7d4bedac2404da47e3c17d", "0fae768fa92047e9955873cce9948106", "75079c1cbdd94ee2864c605e0d87924b", "a28b894d16724777af1108832696562f", "c9334cc2b17044f5a47a946a5ffd6978", "c06b6c1ece3349acaccaa39c001d0153", "1b08d5c58a7f4aeda7eb08d55653e947", "862be556a0854daf960263e703c524da", "8e94e7f4cdfc48e79c432788293be3bd", "97cacb8b41ad40f69389dd743973001e", "eac8cc34255a489c91924e81c232bd9d", "0dfe106624d34aea8d60445c3ab6cf43", "5828aeccd2024a199b9d5771a3977b93", "a2476551203645fa9e42f476ffb97a40", "9f2a214315044f0a966794364101b5fe", "b80f49556dfd42808c4a1731c635636a", "7b82d85cfdf149719f8b33fa08a6319e", "aceb511c29c34504bb2fad6edacd77dd", "4ce0aad5429d41dbb2466205ea9a07aa", "6869f4b8f4fe4046ad84c2e9e2b55df9", "ca1ed6ee9538498ca9c6a81e0df08c7e", "80bef51228db4d10ad922e0aa4e70bfe", "183f5f22e5234476a0ffc797c59dc9d5", "83e63cdb466846d69ffd51bdf070bfb8", "d4b6c18242d14ad38ab6bc1dcf7439b7", "ae581d7619fa4fdbb767bcb7c181b5fc", "fedb4d0db3fc4b969c47ce30656cf3df", "f81216b335c44d48b7f9885a934aef81", "902e2bc5d95e4e6fa412d016dd2afc78", "cb405c27d47045a9856a28d33928de7a", "e9cb1cc7a3aa4f24b161383684e885ef", "c74a58f448d94fd5ac57df0e5115e944", "11d83c357e9a4a499bf8febe54cce845", "6a96e9d2499a41a78f670c62dd04a965", "b04f440665a44ac79ef4f01f239992dc", "acfe086f5b7c4e98b682e95bea0d45e6", "aea477b9eb9540bd8cdc91708c3de0cb", "458b2fc694364652a2b1c0f3306edf2b", "31f0654d87db436f8e7456ba545ed6fa", "0ef0859ac635474c82af31755e27d2a1", "5dc01516dcdc4090bbff6f1a3863047c", "6ef1518ae8434b60a6f50c7be76e508b", "83f21cfc658048b8acbe2a9b906e8af0"]} id="4675e67a" outputId="1988c502-1dbb-4f16-d8c6-1d0b752ac9d0"
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



# %% [markdown] id="badaa5a8"
# ### 1.4 - TikToken
#
# `tiktoken` is a production-ready library which offers high‑speed tokenization used by OpenAI models.  
# Let's compare the older **gpt2** encoding with the newer **cl100k_base** used in GPT‑4.

# %% colab={"base_uri": "https://localhost:8080/"} id="7704c470" outputId="d25ed0b2-fa22-496a-906e-3ff4f2765793"
# Use gpt2 and cl100k_base to encode and decode the following text
# Refer to https://github.com/openai/tiktoken
import tiktoken

sentence = "The 🌟 star-player scored 40 points!"

"""
YOUR CODE HERE
"""

# %% [markdown] id="5e8c1023"
# Experiment: try new sentences, emojis, code snippets, or other languages. If you are interested, try implementing the BPE algorithm yourself.
#
# ### 1.5 - Key Takeaways
#
# * **Word‑level**: simple but brittle (OOV problems).  
# * **Character‑level**: robust but produces long sequences.  
# * **BPE / Byte‑Level BPE**: middle ground used by most LLMs.  
# * **tiktoken**: shows how production models tokenize with pre‑trained sub‑word vocabularies.

# %% [markdown] id="c2a758ba"
# # 2. What is a Language Model?
#
# At its core, a **language model (LM)** is just a *very large* mathematical function built from many neural-network layers.  
# Given a sequence of tokens `[t₁, t₂, …, tₙ]`, it learns to output a probability for the next token `tₙ₊₁`.
#
#
# Each layer applies a simple operation (matrix multiplication, attention, etc.). Stacking hundreds of these layers lets the model capture patterns and statistical relations from text. The final output is a vector of scores that says, “how likely is each possible token to come next?”
#
# > Think of the whole network as **one gigantic equation** whose parameters were tuned during training to minimize prediction error.
#

# %% [markdown] id="8f0c7399"
#
# ### 2.1 - A Single `Linear` Layer
#
# Before we explore Transformer, let’s start tiny:
#
# * A **Linear layer** performs `y = Wx + b`  
#   * `x` – input vector  
#   * `W` – weight matrix (learned)  
#   * `b` – bias vector (learned)
#
# Although this looks basic, chaining thousands of such linear transforms (with nonlinearities in between) gives neural nets their expressive power.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="e425948a" outputId="81fbd114-4ca5-4666-9c9f-07ad48181acd"
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


# %% colab={"base_uri": "https://localhost:8080/"} id="13e5e225" outputId="a29cbd7c-f0b6-4279-8a23-7be9e2eaf978"
import torch.nn as nn, torch

lin = nn.Linear(3, 2)
x = torch.tensor([1.0, -1.0, 0.5])
print("Input :", x)
print("Weights:", lin.weight)
print("Bias   :", lin.bias)
print("Output :", lin(x))


# %% [markdown] id="a04f56bf"
# ### 2.2 - A `Transformer` Layer
#
# Most LLMs are a **stack of identical Transformer blocks**. Each block fuses two main components:
#
# | Step | What it does | Where it lives in code |
# |------|--------------|------------------------|
# | **Multi-Head Self-Attention** | Every token looks at every other token and decides *what matters*. | `block.attn` |
# | **Feed-Forward Network (MLP)** | Re-mixes information token-by-token. | `block.mlp` |
#
# Below, we load the smallest public GPT-2 (124 M parameters), grab its *first* block, and inspect the pieces.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 208, "referenced_widgets": ["37af1deba72242419c85f58b25014673", "18274f2751344c0687fae81ba286f930", "dada5f624514427a9efd705c356e0d34", "d40a6d6e07e640a4ac73020c17067848", "c053b98603f440258b09f6b1e349252c", "b5ffdd2d4bbb462bb3e1ad9f86ae3077", "9bd0b1394e0741ed99dde3a484ed9ad2", "5402052bfb444d07a4a1eba73c67c406", "f43317868cb94131b29caae242ccd643", "8f23fc988e044cac8005332eb9a12339", "78d654682275407ba55fb8a05503b40e", "6d96765e091f4e1d9b3ddba77d1b671f", "690d11a9e3924b06b672b510c5122415", "ff6291958a32426586215ac3e9399ef2", "d9ca4deb36cf4dea9cf55c60521d7583", "48a3a1095e2e42a18ad7dabba4ee6e41", "9e734e5bfaec4716b8e68ba535d12390", "674d8a53b0ef45d8be542615fe3fb74e", "86d7d26bd695435f99c67d5773837149", "6f5c16ca0c844686b50b3bc8a0d6a589", "c80412ea8c5142f392eb74b8cfc80343", "9cbc50826934470a829d4c2a05271905"]} id="47c87f6e" outputId="a9f222e6-da60-4bbb-d652-7b246a863236"
import torch
from transformers import GPT2LMHeadModel

# Load the 124 M-parameter GPT-2 and inspect its layers (12 layers)
"""
YOUR CODE HERE
"""

# %% colab={"base_uri": "https://localhost:8080/"} id="92df06df" outputId="06130c23-2807-4f09-9d77-cbc8be2461b4"
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

# %% [markdown] id="8493ecc8"
# ### 2.3 - Inside GPT-2
#
# GPT-2 is just many of those modules arranged in a repeating *block*. Let's print the modules inside the Transformer.

# %% colab={"base_uri": "https://localhost:8080/"} id="a78ddee1" outputId="7dbb0ccc-91dd-4bb4-a629-c22cd6b6899c"
# Print the name and modules inside gpt2
"""
YOUR CODE HERE
"""

# %% [markdown] id="ed029847"
# As you can see, the Transformer holds various modules, arranged from a list of blocks (`h`). The following table summarizes these modules:
#
# | Step | What it does | Why it matters |
# |------|--------------|----------------|
# | **Token → Embedding** | Converts IDs to vectors | Gives the model a numeric “handle” on words |
# | **Positional Encoding** | Adds “where am I?” info | Order matters in language |
# | **Multi-Head Self-Attention** | Each token asks “which other tokens should I look at?” | Lets the model relate words across a sentence |
# | **Feed-Forward Network** | Two stacked Linear layers with a non-linearity | Mixes information and adds depth |
# | **LayerNorm & Residual** | Stabilize training and help gradients flow | Keeps very deep networks trainable |
#

# %% [markdown] id="0a6a7495"
# ### 2.4 LLM's output
#
# Passing a token sequence through an **LLM** yields a tensor of **logits** with shape  
# `(batch_size, seq_len, vocab_size)`.  
# Applying `softmax` on the last dimension turns those logits into probabilities.
#
# The cell below feeds an 8-token dummy sequence, prints the logits shape, and shows the five most likely next tokens for the final position.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="f98b7b34" outputId="dab929cf-cefc-4d5b-f092-868392f9b1b8"
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


# %% [markdown] id="0eb05c9b"
# ### 2.5 - Key Takeaway
#
# A language model is nothing mystical: it’s a *huge composition* of small, understandable layers trained to predict the next token in a sequence of tokens.

# %% [markdown] id="e0ccf391"
# # 3 - Generation
# Once an LLM is trained to predict the probabilities, we can generate text from the model. This process is called decoding or sampling.
#
# At each step, the LLM outputs a **probability distribution** over the next token. It is the job of the decoding algorithm to pick the next token, and move on to the next token. There are different decoding algorithms and hyper-parameters to control the generaiton:
# * **Greedy** → pick the single highest‑probability token each step (safe but repetitive).  
# * **Top‑k / Nucleus (top‑p)** → sample from a subset of likely tokens (adds variety).
# * **beam** -> applies beam search to pick tokens
# * **Temperature** → a *creativity* knob. Higher values flatten the probability distribution.

# %% [markdown] id="ac0c5728"
# ### 3.1 - Greedy decoding

# %% colab={"base_uri": "https://localhost:8080/"} id="2f2cb953" outputId="6295e944-7ce8-44bf-eb60-6d7878c503ff"
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



# %% colab={"base_uri": "https://localhost:8080/"} id="dbe777ba" outputId="8d0cbc16-078a-47cb-a411-e789b3f3db61"
tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n== GPT-2 | Greedy ==")
    print(generate("gpt2", prompt, "greedy", 80))


# %% [markdown] id="f51d44b2"
#
# Naively picking the single best token every time has the following issues in practice:
#
# * **Loop**: “The cat is is is…”  
# * **Miss long-term payoff**: the highest-probability word *now* might paint you into a boring corner later.

# %% [markdown] id="91607661"
# ### 3.2 - Top-k or top-p sampling

# %% colab={"base_uri": "https://localhost:8080/"} id="0633d4a3" outputId="4d304dc1-4537-4ff6-9e7d-7566c411a815"

tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n== GPT-2 | Top-p ==")
    print(generate("gpt2", prompt, "top-p", 40))


# %% [markdown] id="004b4039"
# ### 3.3 - Try It Yourself
#
# 1. Scroll to the list called `tests`.
# 2. Swap in your own prompts or tweak the decoding strategy.  
# 3. Re‑run the cell and compare the vibes.
#
# > **Tip:** Try the same prompt with `greedy` vs. `top_p` (0.9) and see how the tone changes. Notice especially how small temperature tweaks can soften or sharpen the prose.
#
# * `strategy`: `"greedy"`, `"beam"`, `"top_k"`, `"top_p"`  
# * `temperature`: `0.2 – 2.0`  
# * `k` or `p` thresholds
#
#

# %% [markdown] id="6b775b02"
# # 4 - Completion vs. Instruction-tuned LLMs
#
# We have seen that we can use GPT2 model to pass an input text and generate a new text. However, this model only continues the provided text. It is not engaging in a dialouge-like conversation and cannot be helpful by answering instructions. On the other hand, **instruction-tuned LLMs** like `Qwen-Chat` go through an extra training stage called **post-training** after the base “completion” model is finished. Because of post-training step, an instruction-tuned LLM will:
#
# * **Read the entire prompt as a request,** not just as text to mimic.  
# * **Stay in dialogue mode**. Answer questions, follow steps, ask clarifying queries.  
# * **Refuse or safe-complete** when instructions are unsafe or disallowed.  
# * **Adopt a consistent persona** (e.g., “Assistant”) rather than drifting into story continuation.
#

# %% [markdown] id="1706dc08"
# ### 4.1 - Qwen1.5-8B vs. GPT2
#
# In the code below we’ll feed the same prompt to:
#
# * **GPT-2 (completion-only)** – it will simply keep writing in the same style.  
# * **Qwen-Chat (instruction-tuned)** – it will obey the instruction and respond directly.
#
# Comparing the two outputs makes the difference easy to see.

# %% colab={"base_uri": "https://localhost:8080/", "height": 350, "referenced_widgets": ["9e11091a8f6847f3862648c0fa9bbe03", "4610258527664ff49322d2b60fda1212", "b1423de71d784b2baf183c672e59b882", "2ab201cb663447eeb7b6c45b7f1c8c5f", "53fba4de7a1a43f19ea8bf97658ddecb", "6a623e57be7443458e6f632b3da33cea", "2a9580da37374e609ded703e1198ec34", "d07abec9e2ed447f9391f1fc6a71bd0e", "69f8c8d400a04675b5512a8f4c41e3f4", "b8ca3ca8f57a4cb4babf3dae0b539bb6", "d5b09f9fcb3a452c984837dc281e632d", "720c7f04c39e44d7a70898d5f3559639", "656fd197fe4f41fd882fa33a0b5a65d7", "77edd337adbb457384055160d1b34044", "904ff783e7654411a0c55831616e1fdd", "0d89c18387e64332a9ea7ce7773ae011", "7e80ed38950f497692d40c65629a6a92", "87a0e21d5f2e4396a33b645247b2cebf", "cb9368dbba024bde9822cfffc09bff2a", "6bb292ab0a154f9b867322c9c9338b18", "ab0f6a0bf6804da68fdd34466a04877e", "5652430293b142dba3bab7b67d686f1f", "e774fba3c0fa4f64be7820f887ea6079", "41df5300a9b2421491925e2a876a10ae", "8a41f76668ac474c85eb1aed21e779b2", "c3024556c64842868c122a9a34130cf0", "3b9347f81aa044ea8e227558dbfc62c1", "1f8dbe5fef0644eb9d6916ee32cd8520", "94aca0ee19df40af86da9fead145f8e4", "b4d8d9d47b59410c810fc1602360494c", "98ee5361f5ce45d7bf8f4df7216cd36a", "285410bb02864ba2b2448a45fcf60dd4", "24b19f7b43f14da99831bb1c0a32b127", "281c04259d434a938cd78f6e9038946e", "f245ea14db064cbc90caa686327cbdce", "cae75a5b93bb45e88cdabdcf16078473", "73f9ea0b0ef14194876f068d3cec6541", "381302c0b43147c8b2862086a892dfd7", "97fa61bf53a6491f8df1cc68b58fc7b7", "c19fa1ca20c04776a438401f17a14f1e", "d5708e14b1ef446a8afba270a672a2bc", "279bcc566cd34fa1a3f673ce807f3603", "76564c8e1d42403c8df2e98a682c8c34", "04dfc2b7deb84589b4d0999e1246406b", "026adb7c25a94d508d4f00c999397218", "ce168bd62d1f4a0fad10c3d0f0ba7082", "0845cb6d981344a1b68296a277d93b07", "e339780347d747c8ad2f4ff2fc64ecc6", "ca807ff3ca504b7d9cf941d8242dfe0f", "486a3da3e706453fbb142bb5529f7e3d", "bba42456265a4b4a81c98a767ed7ff3e", "9e40306a6d734fc787c8aa058a354cf0", "85d2c3dac2944f549cb449648e66f412", "53a927c7eebe440ea3649a37ce70bbda", "fd7aa39d378444a290b8690eec35b646", "4dcab84591344812966960d74e49fc02", "cf4a253fa0604874b0596848ee40f071", "f91402cb40a843cda5c511a02504609f", "94e4b13f21264beb9500d6ebf4534a1d", "489a8316340b427591de22aa9e39f969", "6461fe5dde9f48e69db65af01a77497b", "ac84082e9948478c8a071d67c8588cab", "d708f7f766474ac9b187789b48a3c2e1", "a2a947af829e4fc395b1259c992fbcc3", "029482e82e504d0db445aee3ce542cae", "11b19e6f171547e1bc47eb201c08aaa1", "28bb10bc246a4bf2bd30cf445f411f83", "af0e423e449740238a88579abc004442", "23e277cc9cf744efbd6a86e321454743", "e64eee9318644362ba097a3b577fdeb1", "51f42209c88844b785cf942a594edf2f", "ac29b40b61c843d98f7b75d95f48b5f2", "182981eff1e344bbbe24ddedb1a8662a", "c8554bdb04914848850163c0cea75e54", "13f886aeccc34748a0abdb76025317bb", "43f6493a64da410c949b8bd472a0908d", "98bde7609ce34fbaa19d12d2dc83db03"]} id="57b73e7a" outputId="ae830561-bd44-4bb4-93b7-c09c171f6bad"
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


# %% [markdown] id="ef49ab1b"
#
# We downloaded two tiny checkpoints: `GPT‑2` (124 M parameters) and `Qwen‑1.5‑Chat` (1.8 B). If the cell took a while, that was mostly network time. Models are stored locally after the first run.
#
# Let's now generate text and compare two models.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="0c78a508" outputId="f1a64193-ce21-497d-b9a9-24f936d99d79"

tests=[("Once upon a time","greedy"),("What is 2+2?","top_k"),("Suggest a party theme.","top_p")]
for prompt,strategy in tests:
    for key in ["gpt2","qwen"]:
        print(f"\n== {key.upper()} | {strategy} ==")
        print(generate(key,prompt,strategy,80))


# %% [markdown] id="8e1c3da1"
# # 5. (Optional) A Small LLM Playground

# %% [markdown] id="313ba974"
# ### 5.1 ‑ Interactive Playground
#
# Enter a prompt, pick a model and decoding strategy, adjust the temperature, and press **Generate** to watch the model respond.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 339, "referenced_widgets": ["4d7b999f47114af7aef63acff507926a", "53b1a505d202400e9b17487740287505", "3c478b92cefd485daf4230b64d8fea0b", "349ec54e5585460c8a84b4f372172ffd", "5edd4226a0d34e139c5444075c27d378", "070484268e044302932fbc2845c249ad", "d783a353576e4007a6c01867a54f6bdf", "2690d3b55a2b496b8ace0272f3e15075", "f684bdf6d9244e3ab9a2d0a769264def", "c7dd6f0b07524628b18f3f0928e2a94e", "54b2d6d345504e1fac52156273fd8789", "7739f3dd46244714bb2f6f9b1a00cdc5", "bebf97b92969407b934fcb1e3b387c4e", "ef3c998c471a4508878f8c52157afd8a", "9cdd396fb10c4c6bb23f67309dce9e09", "645eb181286c42b2a1d2259f4b84ad2f", "d30fffa79b6a433d84b8718ecc2ff7cf", "73e197d0141b4ec4884186a8dfbaea8b", "74fdb22ba4f740e588ceadcfd09e8999", "f61957b28d4349269bd4e8871e982ba4", "b02b42f4783d4fb384d7dad6d38f8644"]} id="1a67a884" outputId="4003bc26-a3c9-4e22-e58d-ae79e4c2c48b"
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


# %% [markdown] id="cfbccead"
#
# ## 🎉 Congratulations!
#
# You’ve just learned, explored, and inspected a real **LLM**. In one project you:
# * Learned how **tokenization** works in practice
# * Used `tiktoken` library to load and experiment with most advanced tokenizers.
# * Explored LLM architecture and inspected GPT2 blocks and layers
# * Learned decoding strategies and used `top-p` to generate text from GPT2
# * Loaded a powerful chat model, `Qwen1.5-8B` and generated text
# * Built an LLM playground
#
#
# 👏 **Great job!** Take a moment to celebrate. You now have a working mental model of how LLMs work. The skills you used here power most LLMs you see everywhere.
#

# %% [markdown] id="30824bd6"
#
